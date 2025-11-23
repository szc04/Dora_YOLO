// detector_node/src/lib.rs
use dora_node_api::{DoraNode, Event, IntoArrow};
use dora_node_api::dora_core::config::DataId;
use dora_node_api::into_vec;

use tract_onnx::prelude::*;
use tract_ndarray;
use bytemuck;

const MODEL_PATH: &str = "models/yolov8n.onnx";
const INPUT_WIDTH: usize = 640;
const INPUT_HEIGHT: usize = 640;
const CONF_THRESHOLD: f32 = 0.4;
const NMS_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    conf: f32,
    class_id: u32,
}

fn yolo_postprocess(output: &[f32], img_w: u32, img_h: u32) -> Vec<Detection> {
    let num_boxes = 8400;
    let mut boxes = Vec::new();

    for i in 0..num_boxes {
        let obj_conf = output[4 * num_boxes + i];
        if obj_conf < CONF_THRESHOLD {
            continue;
        }

        let mut class_id = 0;
        let mut max_class_score = 0.0;
        for c in 0..80 {
            let score = output[(4 + 1 + c) * num_boxes + i];
            if score > max_class_score {
                max_class_score = score;
                class_id = c as u32;
            }
        }

        let conf = obj_conf * max_class_score;
        if conf < CONF_THRESHOLD {
            continue;
        }

        let cx = output[0 * num_boxes + i] * img_w as f32;
        let cy = output[1 * num_boxes + i] * img_h as f32;
        let w = output[2 * num_boxes + i] * img_w as f32;
        let h = output[3 * num_boxes + i] * img_h as f32;

        let x1 = (cx - w / 2.0).max(0.0);
        let y1 = (cy - h / 2.0).max(0.0);
        let x2 = (cx + w / 2.0).min(img_w as f32);
        let y2 = (cy + h / 2.0).min(img_h as f32);

        boxes.push(Detection { x1, y1, x2, y2, conf, class_id });
    }

    // NMS处理
    boxes.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap());
    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];
    for i in 0..boxes.len() {
        if suppressed[i] { continue; }
        keep.push(i);
        for j in (i + 1)..boxes.len() {
            if suppressed[j] { continue; }
            let inter_x1 = boxes[i].x1.max(boxes[j].x1);
            let inter_y1 = boxes[i].y1.max(boxes[j].y1);
            let inter_x2 = boxes[i].x2.min(boxes[j].x2);
            let inter_y2 = boxes[i].y2.min(boxes[j].y2);
            if inter_x1 >= inter_x2 || inter_y1 >= inter_y2 {
                continue;
            }
            let inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
            let area_i = (boxes[i].x2 - boxes[i].x1) * (boxes[i].y2 - boxes[i].y1);
            let area_j = (boxes[j].x2 - boxes[j].x1) * (boxes[j].y2 - boxes[j].y1);
            let iou = inter_area / (area_i + area_j - inter_area);
            if iou > NMS_THRESHOLD {
                suppressed[j] = true;
            }
        }
    }

    keep.into_iter().map(|i| boxes[i].clone()).collect()
}

#[no_mangle]
pub fn dora_node_main() {
    std::thread::sleep(std::time::Duration::from_millis(500));
    
    let model = match tract_onnx::onnx()
        .model_for_path(MODEL_PATH)
        .and_then(|m| m.with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, INPUT_HEIGHT, INPUT_WIDTH))))
        .and_then(|m| m.into_optimized())
        .and_then(|m| m.into_runnable()) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Detector node: Failed to load ONNX model: {}", e);
                return;
            }
        };

    let (mut node, mut event_stream) = match DoraNode::init_from_env() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Detector node: Failed to initialize DoraNode: {}", e);
            return;
        }
    };
    
    while let Some(event) = event_stream.recv() {
        match event {
            Event::Input { id, data, metadata: _ } if id.as_str() == "frame" => {
                let data_bytes: Vec<u8> = match into_vec::<u8>(&data) {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        eprintln!("Detector node: Failed to convert ArrowData to bytes: {}", e);
                        continue;
                    }
                };

                // 图像格式转换 (HWC BGR u8 -> CHW RGB f32 /255) - 简化处理
                let mut input_array = vec![0.0f32; 3 * INPUT_HEIGHT * INPUT_WIDTH];
                if data_bytes.len() >= 3 * 480 * 640 {  // 确保数据长度足够
                    for (i, pixel) in data_bytes.chunks_exact(3).take(INPUT_HEIGHT * INPUT_WIDTH).enumerate() {
                        let (h, w) = (i / INPUT_WIDTH, i % INPUT_WIDTH);
                        input_array[0 * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w] = pixel[2] as f32 / 255.0; // B->R
                        input_array[1 * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w] = pixel[1] as f32 / 255.0; // G
                        input_array[2 * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w] = pixel[0] as f32 / 255.0; // R->B
                    }
                } else {
                    eprintln!("Detector node: Insufficient image data");
                    continue;
                }

                let arr = match tract_ndarray::Array4::from_shape_vec(
                    (1, 3, INPUT_HEIGHT, INPUT_WIDTH), 
                    input_array
                ) {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("Detector node: Failed to create array: {}", e);
                        continue;
                    }
                };
                
                let input = arr.into_tensor();

                let result = match model.run(tvec![input.into()]) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("Detector node: Model inference failed: {}", e);
                        continue;
                    }
                };
                
                let output_tensor = result[0].clone().into_tensor();
                let output = match output_tensor.to_array_view::<f32>() {
                    Ok(o) => o,
                    Err(e) => {
                        eprintln!("Detector node: Failed to get output tensor as array: {}", e);
                        continue;
                    }
                };

                let output_slice = match output.as_slice() {
                    Some(s) => s,
                    None => {
                        eprintln!("Detector node: Failed to get output tensor as slice");
                        continue;
                    }
                };
                
                let detections = yolo_postprocess(output_slice, 640, 480);

                let mut det_bytes = Vec::new();
                for d in &detections {
                    det_bytes.extend_from_slice(bytemuck::bytes_of(&d.x1));
                    det_bytes.extend_from_slice(bytemuck::bytes_of(&d.y1));
                    det_bytes.extend_from_slice(bytemuck::bytes_of(&d.x2));
                    det_bytes.extend_from_slice(bytemuck::bytes_of(&d.y2));
                    det_bytes.extend_from_slice(bytemuck::bytes_of(&d.conf));
                    det_bytes.extend_from_slice(bytemuck::bytes_of(&d.class_id));
                }

                if let Err(e) = node.send_output(DataId::from("detections".to_string()), Default::default(), det_bytes.into_arrow()) {
                    eprintln!("Detector node: Failed to send detections: {}", e);
                    break;
                }
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
}

