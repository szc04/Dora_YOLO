use dora_node_api::{DoraNode, Event};
use dora_node_api::arrow::array::{UInt8Array, Array};
use opencv::{
    core::{Mat, Scalar, Point, Rect, CV_8UC3},
    highgui,
    imgproc,
    prelude::MatTrait,
};
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
    println!("Visualizer node: Starting...");
    
    let (_node, mut event_stream) = match DoraNode::init_from_env() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Visualizer node: Failed to initialize DoraNode: {}", e);
            std::process::exit(1);
        }
    };

    println!("Visualizer node: Dora node initialized successfully");

    // 存储最新检测结果
    let mut last_detections: Vec<Detection> = Vec::new();
    let _frame_counter = 0;

    // 尝试创建OpenCV窗口
    if highgui::named_window("Visualizer - Camera Feed with Detections", highgui::WINDOW_AUTOSIZE).is_ok() {
        println!("Visualizer node: Display window created successfully");
    } else {
        println!("Visualizer node: Display window creation failed (headless environment?)");
        // 即使窗口创建失败，也要继续运行
    }

    println!("Visualizer node: Ready to receive data");

    loop {
        if let Some(event) = event_stream.recv_timeout(Duration::from_millis(1000)) {
            match event {
                Event::Input { id, data, metadata: _ } => {
                    match id.as_str() {
                        "frame" => {
                            // 处理帧数据
                            println!("Visualizer node: Processing frame input with id 'frame'");
                            
                            // 获取数据类型和长度
                            let array = data.as_any().downcast_ref::<UInt8Array>().expect("Expected UInt8Array");
                            let data_type = array.data_type();
                            let data_length = array.len();
                            
                            println!("Visualizer node: Data type: {:?}", data_type);
                            println!("Visualizer node: Data length: {}", data_length);
                            
                            // 将数据转换为字节向量
                            let img_data: Vec<u8> = array.iter().filter_map(|x| x).collect();
                            println!("Visualizer node: Received frame with {} bytes", img_data.len());
                            
                            // 检查数据长度并创建OpenCV Mat
                            let (width, height) = if img_data.len() == 1280 * 720 * 3 {
                                (1280, 720)
                            } else if img_data.len() == 640 * 480 * 3 {
                                (640, 480)
                            } else if img_data.len() == 320 * 240 * 3 {
                                (320, 240)
                            } else {
                                // 尝试其他可能的分辨率
                                if img_data.len() % 3 == 0 {
                                    // 假设是RGB格式，尝试找到合适的分辨率
                                    let total_pixels = img_data.len() / 3;
                                    // 尝试常见的宽高比
                                    if total_pixels == 1920 * 1080 {
                                        (1920, 1080)
                                    } else if total_pixels == 800 * 600 {
                                        (800, 600)
                                    } else if total_pixels == 320 * 240 {
                                        (320, 240)
                                    } else {
                                        // 默认使用640x480
                                        (640, 480)
                                    }
                                } else {
                                    // 单通道数据
                                    let total_pixels = img_data.len();
                                    if total_pixels == 1280 * 720 {
                                        // 灰度图，转换为RGB
                                        let mut rgb_data = Vec::with_capacity(total_pixels * 3);
                                        for &val in &img_data {
                                            rgb_data.push(val); // R
                                            rgb_data.push(val); // G
                                            rgb_data.push(val); // B
                                        }
                                        println!("Visualizer node: Converted grayscale to RGB, new size: {}", rgb_data.len());
                                        (1280, 720)
                                    } else if total_pixels == 640 * 480 {
                                        // 灰度图，转换为RGB
                                        let mut rgb_data = Vec::with_capacity(total_pixels * 3);
                                        for &val in &img_data {
                                            rgb_data.push(val); // R
                                            rgb_data.push(val); // G
                                            rgb_data.push(val); // B
                                        }
                                        println!("Visualizer node: Converted grayscale to RGB, new size: {}", rgb_data.len());
                                        (640, 480)
                                    } else {
                                        eprintln!("Visualizer node: Unsupported image size: {}", img_data.len());
                                        continue;
                                    }
                                }
                            };
                            
                            // 创建一个空的Mat，然后使用memcpy将数据复制进去
                            let mut mat = unsafe {
                                Mat::new_rows_cols(height as i32, width as i32, CV_8UC3).unwrap()
                            };
                            
                            // 将图像数据复制到Mat中
                            unsafe {
                                let data_ptr = mat.data_mut();
                                std::ptr::copy_nonoverlapping(img_data.as_ptr(), data_ptr, img_data.len());
                            }
                            
                            // 在图像上绘制检测框
                            for detection in &last_detections {
                                // 将相对坐标转换为绝对坐标
                                let x = (detection.x * width as f32) as i32;
                                let y = (detection.y * height as f32) as i32;
                                let w = (detection.width * width as f32) as i32;
                                let h = (detection.height * height as f32) as i32;
                                
                                // 创建检测框
                                let rect = Rect::new(x, y, w, h);
                                
                                // 选择颜色（根据类别）
                                let color = if detection.class_id == 0 {
                                    Scalar::new(0.0, 255.0, 0.0, 0.0) // 绿色 - 人
                                } else if detection.class_id == 1 {
                                    Scalar::new(0.0, 0.0, 255.0, 0.0) // 红色 - 车
                                } else {
                                    Scalar::new(255.0, 0.0, 0.0, 0.0) // 蓝色 - 其他
                                };
                                
                                // 绘制矩形框
                                imgproc::rectangle(
                                    &mut mat,
                                    rect,
                                    color,
                                    2, // 线宽
                                    imgproc::LINE_8,
                                    0, // 线条类型
                                ).unwrap();
                                
                                // 添加标签和置信度
                                let label = format!("{}: {:.2}", detection.class_id, detection.confidence);
                                let org = Point::new(x, y - 10);
                                
                                imgproc::put_text(
                                    &mut mat,
                                    &label,
                                    org,
                                    imgproc::FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    1,
                                    imgproc::LINE_AA,
                                    false,
                                ).unwrap();
                            }
                            
                            // 显示图像
                            if highgui::imshow("Visualizer - Camera Feed with Detections", &mat).is_ok() {
                                // 检查按键事件 (按q或ESC退出)
                                let key = highgui::wait_key(1).unwrap_or(0);
                                if key == 'q' as i32 || key == 27 { // 'q'键或ESC键退出
                                    println!("Visualizer node: Quit key pressed, stopping...");
                                    break;
                                }
                            } else {
                                println!("Visualizer node: Failed to display image");
                            }
                            
                            println!("Visualizer node: Frame displayed with {} detections", last_detections.len());
                        },
                        "detections" => {
                            // 处理检测结果
                            println!("Visualizer node: Processing detections input with id 'detections'");
                            
                            // 解析检测结果
                            if let Some(array) = data.as_any().downcast_ref::<UInt8Array>() {
                                let detection_data: Vec<u8> = array.iter().filter_map(|x| x).collect();
                                println!("Visualizer node: Received {} bytes of detection data", detection_data.len());
                                
                                // 解析检测数据 (假设格式为 [class_id(8字节), confidence(4字节), x(4字节), y(4字节), width(4字节), height(4字节)] 重复)
                                let detection_size = 8 + 4 + 4 + 4 + 4 + 4; // 28字节每检测
                                if detection_data.len() % detection_size == 0 {
                                    last_detections.clear();
                                    
                                    for chunk in detection_data.chunks(detection_size) {
                                        if chunk.len() == detection_size {
                                            let class_id = i64::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]]);
                                            let confidence = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                                            let x = f32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]);
                                            let y = f32::from_le_bytes([chunk[16], chunk[17], chunk[18], chunk[19]]);
                                            let width = f32::from_le_bytes([chunk[20], chunk[21], chunk[22], chunk[23]]);
                                            let height = f32::from_le_bytes([chunk[24], chunk[25], chunk[26], chunk[27]]);
                                            
                                            last_detections.push(Detection {
                                                class_id,
                                                confidence,
                                                x,
                                                y,
                                                width,
                                                height,
                                            });
                                        }
                                    }
                                    
                                    println!("Visualizer node: Parsed {} detections", last_detections.len());
                                } else {
                                    eprintln!("Visualizer node: Invalid detection data size: {}", detection_data.len());
                                }
                            }
                        },
                        _ => {
                            println!("Visualizer node: Received input with id '{}', ignoring", id);
                        }
                    }
                }
                Event::Stop(_) => {
                    println!("Visualizer node: Received stop event");
                    break;
                }
                Event::Error(e) => {
                    println!("Visualizer node: Received error event: {}", e);
                    continue; // 继续运行
                }
                _ => {
                    println!("Visualizer node: Received other event: {:?}", event);
                }
            }
        }
    }

    // 销毁窗口
    highgui::destroy_all_windows().unwrap();
    println!("Visualizer node: Finished");
}

