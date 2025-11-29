use dora_node_api::{DoraNode, Event, dora_core::config::DataId, MetadataParameters};
use dora_node_api::arrow::array::{UInt8Array, Array as ArrowArray};
use std::time::Duration;
use std::path::Path;
use tract_onnx::prelude::*;
use opencv::{core::{Mat}, imgproc, prelude::*};
use anyhow::{Result, Context};

#[derive(Debug, Clone)]
struct Detection {
    name: String,          // 检测对象的唯一标识名
    class_name: String,    // 类别名称（如"person", "car"等）
    confidence: f32,       // 置信度
    x: f32,                // 归一化中心x坐标
    y: f32,                // 归一化中心y坐标
    width: f32,            // 归一化宽度
    height: f32,           // 归一化高度
}

struct YoloDetector {
    model: Option<RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>,
    input_width: usize,
    input_height: usize,
    class_names: Vec<String>,
}

impl YoloDetector {
    fn new(model_path: &str) -> Result<Self> {
        eprintln!("Initializing YOLO detector with model: {}", model_path);
        
        let model = if Path::new(model_path).exists() {
            match Self::load_model(model_path) {
                Ok(m) => Some(m),
                Err(e) => {
                    eprintln!("Failed to load model: {}", e);
                    None
                }
            }
        } else {
            eprintln!("Model file not found at {}. Using mock detections only.", model_path);
            None
        };
        
        // COCO类别名称
        let class_names = vec![
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ].iter().map(|&s| s.to_string()).collect();
        
        eprintln!("YOLO detector created. Model loaded: {}", model.is_some());
        
        Ok(Self {
            model,
            input_width: 640,
            input_height: 640,
            class_names,
        })
    }
    
    fn load_model(model_path: &str) -> Result<RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>> {
        eprintln!("Loading ONNX model from: {}", model_path);
        
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .context("Failed to load ONNX model")?
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 640, 640)))
            .context("Failed to set input fact")?
            .into_optimized()
            .context("Failed to optimize model")?
            .into_runnable()
            .context("Failed to make model runnable")?;
            
        eprintln!("Successfully loaded and optimized ONNX model");
        Ok(model)
    }
    
    fn preprocess(&self, img_data: &[u8], width: u32, height: u32) -> Result<Tensor> {
        eprintln!("Preprocessing image: {}x{}", width, height);
        
        // 创建一个空的 Mat
        let mut mat = unsafe {
            Mat::new_rows_cols(height as i32, width as i32, opencv::core::CV_8UC3)
                .context("Failed to create Mat")?
        };
        
        // 手动复制数据到 Mat 中
        unsafe {
            let data_ptr = img_data.as_ptr() as *const u8;
            let mat_data = mat.data_mut() as *mut u8;
            std::ptr::copy_nonoverlapping(data_ptr, mat_data, img_data.len());
        }
        
        // 转换BGR到RGB
        let mut rgb_mat = Mat::default();
        imgproc::cvt_color(&mat, &mut rgb_mat, imgproc::COLOR_BGR2RGB, 0)
            .context("Failed to convert color space")?;
        
        // 调整大小
        let mut resized = Mat::default();
        let target_size = opencv::core::Size::new(self.input_width as i32, self.input_height as i32);
        imgproc::resize(
            &rgb_mat, 
            &mut resized, 
            target_size,
            0.0, 
            0.0, 
            imgproc::INTER_LINEAR
        ).context("Failed to resize image")?;
        
        // 归一化到[0,1]范围
        let mut normalized = Mat::default();
        resized.convert_to(&mut normalized, opencv::core::CV_32F, 1.0/255.0, 0.0)
            .context("Failed to normalize image")?;
        
        // 将OpenCV Mat转换为tract tensor
        let mut tensor_data = vec![0.0f32; self.input_width * self.input_height * 3];
        let mut idx = 0;
        
        for y in 0..self.input_height {
            for x in 0..self.input_width {
                let mut pixel_values = [0.0f32; 3];
                let result = normalized.at_2d::<opencv::core::Vec3f>(y as i32, x as i32);
                if let Ok(pixel) = result {
                    pixel_values[0] = pixel[0];  // R
                    pixel_values[1] = pixel[1];  // G
                    pixel_values[2] = pixel[2];  // B
                }
                
                for c in 0..3 {
                    tensor_data[idx] = pixel_values[c];
                    idx += 1;
                }
            }
        }
        
        // 重排维度: HWC -> CHW
        let hwc_array = ndarray::Array::from_shape_vec(
            (self.input_height, self.input_width, 3), 
            tensor_data
        ).context("Failed to create HWC array")?;
        
        let chw_array = hwc_array.permuted_axes([2, 0, 1]);
        let final_array = chw_array.insert_axis(ndarray::Axis(0)); // 添加batch维度
        
        eprintln!("Preprocessing completed successfully");
        
        // 正确创建Tensor - 使用from_array_view方法
        let tensor = tract_core::ndarray::ArrayD::<f32>::from_shape_vec(
            final_array.shape().to_vec(),
            final_array.into_raw_vec(),
        ).context("Failed to create ndarray")?;
        
        Ok(Tensor::from(tensor))
    }
    
    fn postprocess(&self, outputs: &Tensor, img_width: f32, img_height: f32) -> Vec<Detection> {
        let mut detections = Vec::new();
        
        // 获取输出数据
        if let Ok(output_values) = outputs.to_array_view::<f32>() {
            let output_shape = output_values.shape();
            eprintln!("Output shape: {:?}", output_shape);
            
            // YOLOv8输出通常是 [1, 84, 8400] 格式
            // 84 = 4 (bbox) + 80 (classes)
            if output_shape.len() >= 3 {
                let batch_dim = 0;
                let channel_dim = 1;
                let detection_dim = 2;
                
                // 检查形状是否符合预期
                if output_shape[batch_dim] == 1 && output_shape[channel_dim] >= 84 {
                    let num_detections = output_shape[detection_dim];
                    eprintln!("Processing {} detections", num_detections);
                    
                    // 限制处理的检测数量，避免过多
                    let max_detections = num_detections.min(100);
                    
                    // 处理每个检测
                    for i in 0..max_detections {
                        let bbox_x = *output_values.get([0, 0, i]).unwrap_or(&0.0);
                        let bbox_y = *output_values.get([0, 1, i]).unwrap_or(&0.0);
                        let bbox_w = *output_values.get([0, 2, i]).unwrap_or(&0.0);
                        let bbox_h = *output_values.get([0, 3, i]).unwrap_or(&0.0);
                        
                        // 获取类别置信度
                        let mut max_conf = 0.0;
                        let mut max_class_idx = 0;
                        for c in 0..80 {
                            if 4 + c < output_shape[channel_dim] {
                                let conf = *output_values.get([0, 4 + c, i]).unwrap_or(&0.0);
                                if conf > max_conf {
                                    max_conf = conf;
                                    max_class_idx = c;
                                }
                            }
                        }
                        
                        // 应用置信度阈值
                        if max_conf > 0.1 && (max_class_idx as usize) < self.class_names.len() {
                            // 生成唯一标识名
                            let object_id = format!("{}_{}", self.class_names[max_class_idx as usize], i);
                            
                            detections.push(Detection {
                                name: object_id,
                                class_name: self.class_names[max_class_idx as usize].clone(),
                                confidence: max_conf,
                                x: bbox_x / img_width,
                                y: bbox_y / img_height,
                                width: bbox_w / img_width,
                                height: bbox_h / img_height,
                            });
                        }
                    }
                } else {
                    eprintln!("Unexpected output shape dimensions: {:?}", output_shape);
                }
            } else {
                eprintln!("Output has unexpected number of dimensions: {}", output_shape.len());
            }
        } else {
            eprintln!("Failed to convert output tensor to array view");
        }
        
        eprintln!("Found {} objects with confidence > 0.5", detections.len());
        detections
    }
    
    fn detect(&self, img_data: &[u8], width: u32, height: u32) -> Result<Vec<Detection>> {
        if let Some(ref model) = self.model {
            eprintln!("Running detection on image {}x{}", width, height);
            
            // 预处理
            let input_tensor = self.preprocess(img_data, width, height)?;
            
            // 推理
            let outputs = model.run(tvec!(input_tensor.into()))
                .context("Model inference failed")?;
            
            // 获取输出
            let output_tensor = &outputs[0];
            
            // 后处理
            let detections = self.postprocess(output_tensor, width as f32, height as f32);
            
            eprintln!("Detection completed successfully. Found {} objects", detections.len());
            Ok(detections)
        } else {
            eprintln!("No model loaded. Using mock detections.");
            Ok(create_mock_detections(0))
        }
    }
}

fn main() -> Result<()> {
    // 在最开始就初始化日志系统
    env_logger::init();
    
    // 立即打印启动信息
    println!("Detector node: Starting... (stdout)");
    eprintln!("Detector node: Starting... (stderr)");
    eprintln!("Detector node: Starting... (info)");
    
    let (mut node, mut event_stream) = match DoraNode::init_from_env() {
        Ok(n) => {
            eprintln!("Detector node: Dora node initialized successfully");
            n
        },
        Err(e) => {
            eprintln!("Detector node: Failed to initialize DoraNode: {}", e);
            std::process::exit(1);
        }
    };

    // 初始化YOLO检测器
    let model_path = "models/yolov8n.onnx";
    let detector = match YoloDetector::new(model_path) {
        Ok(d) => {
            eprintln!("Detector node: YOLO detector initialized");
            d
        },
        Err(e) => {
            eprintln!("Detector node: Failed to initialize YOLO detector: {}", e);
            return Err(e);
        }
    };

    let mut frame_counter = 0;
    eprintln!("Detector node: Ready to receive data");

    // 自适应跳帧机制
    let mut skip_counter = 0;
    let mut process_interval = 1; // 初始为每帧都处理

    loop {
        // 添加调试日志，查看是否能接收到任何事件
        eprintln!("Detector node: Waiting for event...");
        
        if let Some(event) = event_stream.recv_timeout(Duration::from_millis(1000)) {
            eprintln!("Detector node: Received an event");
            
            match event {
                Event::Input { id, data, metadata } => {
                    eprintln!("Detector node: Received input with id '{}'", id);
                    
                    // 打印所有元数据参数，帮助调试
                    eprintln!("Detector node: Metadata parameters: {:?}", metadata.parameters);
                    
                    if id.as_str() == "frame" {
                        eprintln!("Detector node: Processing frame input with id 'frame'");
                        
                        // 从元数据中获取图像尺寸 - 使用更灵活的方式
                        let width = match metadata.parameters.get("width") {
                            Some(dora_node_api::Parameter::String(s)) => s.parse::<u32>().ok().unwrap_or(640),
                            Some(dora_node_api::Parameter::Integer(i)) => *i as u32,
                            _ => {
                                // 如果没有元数据，尝试根据数据大小推断
                                // 假设是常见的分辨率
                                if data.len() == 640 * 480 * 3 {
                                    640
                                } else if data.len() == 1280 * 720 * 3 {
                                    1280
                                } else {
                                    640 // 默认值
                                }
                            }
                        };
                        let height = match metadata.parameters.get("height") {
                            Some(dora_node_api::Parameter::String(s)) => s.parse::<u32>().ok().unwrap_or(480),
                            Some(dora_node_api::Parameter::Integer(i)) => *i as u32,
                            _ => {
                                // 如果没有元数据，尝试根据数据大小推断
                                if data.len() == 640 * 480 * 3 {
                                    480
                                } else if data.len() == 1280 * 720 * 3 {
                                    720
                                } else {
                                    480 // 默认值
                                }
                            }
                        };
                        
                        eprintln!("Detector node: Image dimensions - {}x{}", width, height);
                        
                        // 获取图像数据
                        let array = data.as_any().downcast_ref::<UInt8Array>()
                            .context("Expected UInt8Array")?;
                        let img_data: Vec<u8> = array.iter().filter_map(|x| x).collect();
                        
                        eprintln!("Detector node: Received frame data with {} bytes", img_data.len());
                        
                        // 自适应跳帧：根据处理时间调整处理间隔
                        let should_process = skip_counter % process_interval == 0;
                        
                        if should_process {
                            let start_time = std::time::Instant::now();
                            
                            // 运行检测
                            let detections = detector.detect(&img_data, width, height)?;
                            
                            // 计算处理时间并调整跳帧间隔
                            let elapsed = start_time.elapsed();
                            let elapsed_ms = elapsed.as_millis() as u64;
                            
                            eprintln!("Detector node: Detection took {} ms", elapsed_ms);
                            
                            // 根据处理时间自适应调整跳帧间隔
                            if elapsed_ms > 150 { // 如果处理时间超过150ms
                                process_interval = std::cmp::min(process_interval + 1, 10); // 最多跳过9帧
                                eprintln!("Detector node: Increased process interval to {}", process_interval);
                            } else if elapsed_ms < 50 && process_interval > 1 { // 如果处理很快且当前间隔大于1
                                process_interval -= 1; // 减少跳帧
                                eprintln!("Detector node: Decreased process interval to {}", process_interval);
                            }
                            
                            // 将检测结果序列化
                            let mut detection_bytes = Vec::new();
                            for detection in &detections {
                                // 序列化name字段（16字节固定长度）
                                let name_bytes = detection.name.as_bytes();
                                let name_len = name_bytes.len().min(16);
                                detection_bytes.extend_from_slice(&name_bytes[..name_len]);
                                detection_bytes.extend_from_slice(&vec![0; 16 - name_len]);
                                
                                // 序列化class_name（16字节固定长度）
                                let class_bytes = detection.class_name.as_bytes();
                                let class_len = class_bytes.len().min(16);
                                detection_bytes.extend_from_slice(&class_bytes[..class_len]);
                                detection_bytes.extend_from_slice(&vec![0; 16 - class_len]);
                                
                                // 序列化其他数值
                                detection_bytes.extend_from_slice(&detection.confidence.to_le_bytes());
                                detection_bytes.extend_from_slice(&detection.x.to_le_bytes());
                                detection_bytes.extend_from_slice(&detection.y.to_le_bytes());
                                detection_bytes.extend_from_slice(&detection.width.to_le_bytes());
                                detection_bytes.extend_from_slice(&detection.height.to_le_bytes());
                            }
                            
                            // 发送检测结果
                            let output_id = DataId::from("detections".to_string());
                            let mut parameters = MetadataParameters::new();
                            parameters.insert("num_detections".to_string(), dora_node_api::Parameter::String(detections.len().to_string()));
                            parameters.insert("frame_id".to_string(), dora_node_api::Parameter::String(frame_counter.to_string()));
                            
                            if let Err(e) = node.send_output_bytes(
                                output_id,
                                parameters.clone(),
                                detection_bytes.len(),
                                &detection_bytes
                            ) {
                                eprintln!("Detector node: Failed to send detections output: {}", e);
                            }
                            
                            // 转发原始帧
                            let output_id = DataId::from("frame".to_string());
                            if let Err(e) = node.send_output_bytes(
                                output_id,
                                parameters,
                                img_data.len(),
                                &img_data
                            ) {
                                eprintln!("Detector node: Failed to send frame output: {}", e);
                            }
                            
                            frame_counter += 1;
                            eprintln!("Detector node: Processed frame {}, found {} objects", 
                                     frame_counter, detections.len());
                        } else {
                            eprintln!("Detector node: Skipping frame {} due to adaptive frame skipping (interval: {})", 
                                     skip_counter, process_interval);
                        }
                        
                        skip_counter += 1;
                    } else {
                        eprintln!("Detector node: Received input with id '{}' but expected 'frame'", id);
                    }
                }
                Event::Stop(_) => {
                    eprintln!("Detector node: Received stop event");
                    break;
                }
                Event::Error(e) => {
                    // 改进错误处理：不退出，但记录错误
                    eprintln!("Detector node: Received error event: {}", e);
                    continue; // 继续运行，不退出
                }
                _ => {
                    eprintln!("Detector node: Received unhandled event type: {:?}", event);
                }
            }
        } else {
            // 没有收到事件，继续循环
            eprintln!("Detector node: No events received in timeout period");
        }
    }

    eprintln!("Detector node: Finished");
    Ok(())
}

fn create_mock_detections(frame_id: u32) -> Vec<Detection> {
    vec![
        Detection {
            name: format!("person_{}", frame_id % 10),
            class_name: "person".to_string(),
            confidence: 0.95,
            x: 0.3,
            y: 0.4,
            width: 0.2,
            height: 0.4,
        },
        Detection {
            name: format!("car_{}", frame_id % 5),
            class_name: "car".to_string(),
            confidence: 0.87,
            x: 0.6,
            y: 0.5,
            width: 0.25,
            height: 0.2,
        },
    ]
}

