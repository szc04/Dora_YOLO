use dora_node_api::{DoraNode, Event, dora_core::config::DataId, MetadataParameters};
use dora_node_api::arrow::array::{UInt8Array, Array};
use std::time::Duration;

#[derive(Debug, Clone)]
struct Detection {
    class_id: i64,
    confidence: f32,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

fn main() {
    println!("Detector node: Starting...");
    
    let (mut node, mut event_stream) = match DoraNode::init_from_env() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Detector node: Failed to initialize DoraNode: {}", e);
            std::process::exit(1);
        }
    };

    println!("Detector node: Dora node initialized successfully");

    // 模拟检测结果
    let mut frame_counter = 0;

    println!("Detector node: Ready to receive data");

    loop {
        if let Some(event) = event_stream.recv_timeout(Duration::from_millis(1000)) {
            match event {
                Event::Input { id, data, metadata: _ } => {
                    match id.as_str() {
                        "frame" => {
                            // 处理帧数据
                            println!("Detector node: Processing frame input with id 'frame'");
                            
                            // 获取数据类型和长度
                            let array = data.as_any().downcast_ref::<UInt8Array>().expect("Expected UInt8Array");
                            let data_type = array.data_type();
                            let data_length = array.len();
                            
                            println!("Detector node: Data type: {:?}", data_type);
                            println!("Detector node: Data length: {}", data_length);
                            
                            // 将数据转换为字节向量
                            let img_data: Vec<u8> = array.iter().filter_map(|x| x).collect();
                            println!("Detector node: Received frame with {} bytes", img_data.len());
                            
                            // 检查数据长度是否合理 (1280x720x3 = 2764800)
                            if img_data.len() == 1280 * 720 * 3 {
                                // 模拟检测结果 - 生成一些示例检测框
                                let detections = vec![
                                    Detection {
                                        class_id: 0,  // 人
                                        confidence: 0.95,
                                        x: 0.3,      // 相对坐标
                                        y: 0.4,
                                        width: 0.2,
                                        height: 0.4,
                                    },
                                    Detection {
                                        class_id: 1,  // 车
                                        confidence: 0.87,
                                        x: 0.6,
                                        y: 0.5,
                                        width: 0.25,
                                        height: 0.2,
                                    },
                                ];
                                
                                // 将检测结果打包成字节数组
                                let mut detection_bytes = Vec::new();
                                for detection in &detections {
                                    detection_bytes.extend_from_slice(&detection.class_id.to_le_bytes());
                                    detection_bytes.extend_from_slice(&detection.confidence.to_le_bytes());
                                    detection_bytes.extend_from_slice(&detection.x.to_le_bytes());
                                    detection_bytes.extend_from_slice(&detection.y.to_le_bytes());
                                    detection_bytes.extend_from_slice(&detection.width.to_le_bytes());
                                    detection_bytes.extend_from_slice(&detection.height.to_le_bytes());
                                }
                                
                                // 发送检测结果
                                let output_id = DataId::from("detections".to_string());
                                let parameters = MetadataParameters::default();
                                
                                match node.send_output_bytes(output_id, parameters, detection_bytes.len(), &detection_bytes) {
                                    Ok(_) => {
                                        println!("Detector node: Sent {} detections", detections.len());
                                    },
                                    Err(e) => {
                                        eprintln!("Detector node: Failed to send detections: {}", e);
                                    }
                                }
                                
                                // 也可以将原始帧转发到输出端
                                let output_id = DataId::from("frame".to_string());
                                let parameters = MetadataParameters::default();
                                
                                match node.send_output_bytes(output_id, parameters, img_data.len(), &img_data) {
                                    Ok(_) => {
                                        frame_counter += 1;
                                        println!("Detector node: Forwarded frame {}", frame_counter);
                                    },
                                    Err(e) => {
                                        eprintln!("Detector node: Failed to forward frame: {}", e);
                                    }
                                }
                            } else {
                                // 如果数据长度不匹配，尝试其他常见分辨率
                                if img_data.len() == 640 * 480 * 3 {
                                    // 640x480图像
                                    
                                    // 模拟检测结果
                                    let detections = vec![
                                        Detection {
                                            class_id: 0,  // 人
                                            confidence: 0.92,
                                            x: 0.4,
                                            y: 0.5,
                                            width: 0.2,
                                            height: 0.3,
                                        },
                                    ];
                                    
                                    // 将检测结果打包成字节数组
                                    let mut detection_bytes = Vec::new();
                                    for detection in &detections {
                                        detection_bytes.extend_from_slice(&detection.class_id.to_le_bytes());
                                        detection_bytes.extend_from_slice(&detection.confidence.to_le_bytes());
                                        detection_bytes.extend_from_slice(&detection.x.to_le_bytes());
                                        detection_bytes.extend_from_slice(&detection.y.to_le_bytes());
                                        detection_bytes.extend_from_slice(&detection.width.to_le_bytes());
                                        detection_bytes.extend_from_slice(&detection.height.to_le_bytes());
                                    }
                                    
                                    // 发送检测结果
                                    let output_id = DataId::from("detections".to_string());
                                    let parameters = MetadataParameters::default();
                                    
                                    match node.send_output_bytes(output_id, parameters, detection_bytes.len(), &detection_bytes) {
                                        Ok(_) => {
                                            println!("Detector node: Sent {} detections (640x480)", detections.len());
                                        },
                                        Err(e) => {
                                            eprintln!("Detector node: Failed to send detections: {}", e);
                                        }
                                    }
                                    
                                    // 转发帧
                                    let output_id = DataId::from("frame".to_string());
                                    let parameters = MetadataParameters::default();
                                    
                                    match node.send_output_bytes(output_id, parameters, img_data.len(), &img_data) {
                                        Ok(_) => {
                                            frame_counter += 1;
                                            println!("Detector node: Forwarded frame {} (640x480)", frame_counter);
                                        },
                                        Err(e) => {
                                            eprintln!("Detector node: Failed to forward frame: {}", e);
                                        }
                                    }
                                } else {
                                    // 尝试处理单通道数据
                                    println!("Detector node: Expanding single-channel data ({} bytes) to BGR image", img_data.len());
                                    
                                    // 假设是灰度图像，扩展为BGR
                                    let expected_pixels = 1280 * 720;
                                    if img_data.len() == expected_pixels {
                                        // 创建BGR图像数据
                                        let mut bgr_data = Vec::with_capacity(expected_pixels * 3);
                                        for &gray_val in &img_data {
                                            bgr_data.push(gray_val); // Blue
                                            bgr_data.push(gray_val); // Green
                                            bgr_data.push(gray_val); // Red
                                        }
                                        
                                        // 模拟检测结果
                                        let detections = vec![
                                            Detection {
                                                class_id: 0,  // 人
                                                confidence: 0.88,
                                                x: 0.35,
                                                y: 0.45,
                                                width: 0.15,
                                                height: 0.25,
                                            },
                                        ];
                                        
                                        // 将检测结果打包成字节数组
                                        let mut detection_bytes = Vec::new();
                                        for detection in &detections {
                                            detection_bytes.extend_from_slice(&detection.class_id.to_le_bytes());
                                            detection_bytes.extend_from_slice(&detection.confidence.to_le_bytes());
                                            detection_bytes.extend_from_slice(&detection.x.to_le_bytes());
                                            detection_bytes.extend_from_slice(&detection.y.to_le_bytes());
                                            detection_bytes.extend_from_slice(&detection.width.to_le_bytes());
                                            detection_bytes.extend_from_slice(&detection.height.to_le_bytes());
                                        }
                                        
                                        // 发送检测结果
                                        let output_id = DataId::from("detections".to_string());
                                        let parameters = MetadataParameters::default();
                                        
                                        match node.send_output_bytes(output_id, parameters, detection_bytes.len(), &detection_bytes) {
                                            Ok(_) => {
                                                println!("Detector node: Sent {} detections (converted grayscale)", detections.len());
                                            },
                                            Err(e) => {
                                                eprintln!("Detector node: Failed to send detections: {}", e);
                                            }
                                        }
                                        
                                        // 转发BGR帧
                                        let output_id = DataId::from("frame".to_string());
                                        let parameters = MetadataParameters::default();
                                        
                                        match node.send_output_bytes(output_id, parameters, bgr_data.len(), &bgr_data) {
                                            Ok(_) => {
                                                frame_counter += 1;
                                                println!("Detector node: Forwarded BGR frame {}", frame_counter);
                                            },
                                            Err(e) => {
                                                eprintln!("Detector node: Failed to forward BGR frame: {}", e);
                                            }
                                        }
                                    } else {
                                        eprintln!("Detector node: Unexpected image data size: {}", img_data.len());
                                    }
                                }
                            }
                            
                            println!("Detector node: Frame counter added successfully");
                        },
                        _ => {
                            println!("Detector node: Received input with id '{}', ignoring", id);
                        }
                    }
                }
                Event::Stop(_) => {
                    println!("Detector node: Received stop event");
                    break;
                }
                Event::Error(e) => {
                    println!("Detector node: Received error event: {}", e);
                    continue; // 继续运行
                }
                _ => {
                    println!("Detector node: Received other event: {:?}", event);
                }
            }
        }
    }

    println!("Detector node: Finished");
}

