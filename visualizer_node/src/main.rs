use dora_node_api::{DoraNode, Event};
use dora_node_api::into_vec;

use opencv::{
    core::{self, Mat, Rect, Point, Scalar},
    highgui,
    imgproc,
    prelude::*,
};
use bytemuck;

// COCO 类别名
const CLASS_NAMES: &[&str] = &[
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

#[derive(Debug, Clone)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    conf: f32,
    class_id: u32,
}

fn main() {
    std::thread::sleep(std::time::Duration::from_millis(500));
    
    let (_node, mut event_stream) = match DoraNode::init_from_env() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Visualizer node: Failed to initialize DoraNode: {}", e);
            std::process::exit(1);
        }
    };
    
    let mut frame_cache: Option<Vec<u8>> = None;
    let mut detection_cache: Option<Vec<u8>> = None;

    while let Some(event) = event_stream.recv() {
        match event {
            Event::Input { id, data, metadata: _ } => {
                if id.as_str() == "frame" {
                    match into_vec::<u8>(&data) {
                        Ok(bytes) => frame_cache = Some(bytes),
                        Err(e) => eprintln!("Visualizer node: Failed to convert frame ArrowData to bytes: {}", e),
                    }
                } else if id.as_str() == "detections" {
                    match into_vec::<u8>(&data) {
                        Ok(bytes) => detection_cache = Some(bytes),
                        Err(e) => eprintln!("Visualizer node: Failed to convert detections ArrowData to bytes: {}", e),
                    }
                }

                if let (Some(frame_data), Some(det_data)) = (frame_cache.take(), detection_cache.take()) {
                    let mut frame = match unsafe { Mat::new_rows_cols(480, 640, core::CV_8UC3) } {
                        Ok(mat) => mat,
                        Err(e) => {
                            eprintln!("Visualizer node: Failed to create frame Mat: {}", e);
                            continue;
                        }
                    };
                    
                    match frame.data_typed_mut::<u8>() {
                        Ok(frame_data_mut) => {
                            if frame_data.len() == frame_data_mut.len() {
                                frame_data_mut.copy_from_slice(&frame_data);
                            } else {
                                eprintln!("Visualizer node: Frame data size mismatch");
                                continue;
                            }
                        },
                        Err(e) => {
                            eprintln!("Visualizer node: Failed to get mutable frame data: {}", e);
                            continue;
                        }
                    }

                    let mut detections = Vec::new();
                    for chunk in det_data.chunks_exact(24) {
                        if chunk.len() == 24 {
                            let x1 = match bytemuck::try_from_bytes::<f32>(&chunk[0..4]) {
                                Ok(val) => *val,
                                Err(_) => continue,
                            };
                            let y1 = match bytemuck::try_from_bytes::<f32>(&chunk[4..8]) {
                                Ok(val) => *val,
                                Err(_) => continue,
                            };
                            let x2 = match bytemuck::try_from_bytes::<f32>(&chunk[8..12]) {
                                Ok(val) => *val,
                                Err(_) => continue,
                            };
                            let y2 = match bytemuck::try_from_bytes::<f32>(&chunk[12..16]) {
                                Ok(val) => *val,
                                Err(_) => continue,
                            };
                            let conf = match bytemuck::try_from_bytes::<f32>(&chunk[16..20]) {
                                Ok(val) => *val,
                                Err(_) => continue,
                            };
                            let class_id = match bytemuck::try_from_bytes::<u32>(&chunk[20..24]) {
                                Ok(val) => *val,
                                Err(_) => continue,
                            };

                            detections.push(Detection {
                                x1, y1, x2, y2, conf, class_id
                            });
                        }
                    }

                    for d in detections {
                        if d.class_id < CLASS_NAMES.len() as u32 {
                            let label = format!("{} {:.2}", CLASS_NAMES[d.class_id as usize], d.conf);
                            let rect = Rect::new(
                                d.x1 as i32,
                                d.y1 as i32,
                                (d.x2 - d.x1) as i32,
                                (d.y2 - d.y1) as i32,
                            );
                            
                            if imgproc::rectangle(
                                &mut frame, 
                                rect, 
                                Scalar::new(0.0, 255.0, 0.0, 0.0), 
                                2, 
                                imgproc::LINE_8, 
                                0
                            ).is_ok() {
                                let _ = imgproc::put_text(
                                    &mut frame, 
                                    &label, 
                                    Point::new(d.x1 as i32, (d.y1 - 10.0) as i32),
                                    imgproc::FONT_HERSHEY_SIMPLEX, 
                                    0.5, 
                                    Scalar::new(0.0, 255.0, 0.0, 0.0), 
                                    2, 
                                    imgproc::LINE_AA, 
                                    false
                                );
                            }
                        }
                    }

                    if highgui::imshow("YOLO Rust + Dora", &frame).is_ok() {
                        if highgui::wait_key(1).unwrap_or(0) == 113 { // 'q'键退出
                            break;
                        }
                    }
                }
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    
    let _ = highgui::destroy_all_windows();
    eprintln!("Visualizer node: Finished");
}

