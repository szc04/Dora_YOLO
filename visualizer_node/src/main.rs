use dora_node_api::{DoraNode, Event};
use dora_node_api::arrow::array::{UInt8Array, Array};
use opencv::{
    core::{Mat, Scalar, Point, Rect, CV_8UC3},
    highgui,
    imgproc::{self, LINE_8, LINE_AA, FONT_HERSHEY_SIMPLEX},
    prelude::{MatTraitConst, MatTrait},
};
use std::time::Duration;
use log::{info, warn, error};
use anyhow::{Result, Context};
use std::str;

#[derive(Debug, Clone)]
struct Detection {
    name: String,          // 检测对象的唯一标识名
    class_name: String,    // 类别名称
    confidence: f32,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

// 为不同类别定义颜色
fn get_class_color(class_name: &str) -> Scalar {
    let color_map = [
        ("person", (0.0, 255.0, 0.0)),     // 绿色
        ("car", (0.0, 0.0, 255.0)),        // 红色
        ("truck", (0.0, 0.0, 200.0)),      // 深红色
        ("bus", (0.0, 0.0, 150.0)),        // 更深的红色
        ("motorcycle", (0.0, 150.0, 255.0)), // 橙色
        ("bicycle", (255.0, 150.0, 0.0)),  // 青色
        ("dog", (255.0, 0.0, 255.0)),      // 紫色
        ("cat", (150.0, 0.0, 255.0)),      // 深紫色
    ];
    
    for &(class, (b, g, r)) in &color_map {
        if class_name == class {
            return Scalar::new(b, g, r, 0.0);
        }
    }
    
    // 默认颜色（蓝色）
    Scalar::new(255.0, 0.0, 0.0, 0.0)
}

fn main() -> Result<()> {
    env_logger::init();
    info!("Visualizer node: Starting...");
    
    let (_node, mut event_stream) = match DoraNode::init_from_env() {
        Ok(n) => n,
        Err(e) => {
            error!("Visualizer node: Failed to initialize DoraNode: {}", e);
            std::process::exit(1);
        }
    };
    
    info!("Visualizer node: Dora node initialized successfully");
    
    // 存储最新检测结果
    let mut last_detections: Vec<Detection> = Vec::new();
    let mut frame_counter = 0;
    
    // 尝试创建OpenCV窗口
    if highgui::named_window("Visualizer - Camera Feed with Detections", highgui::WINDOW_AUTOSIZE).is_ok() {
        info!("Visualizer node: Display window created successfully");
    } else {
        warn!("Visualizer node: Display window creation failed (headless environment?)");
        // 即使窗口创建失败，也要继续运行
    }
    
    info!("Visualizer node: Ready to receive data");
    
    loop {
        if let Some(event) = event_stream.recv_timeout(Duration::from_millis(1000)) {
            match event {
                Event::Input { id, data, metadata } => {
                    match id.as_str() {
                        "frame" => {
                            // 处理帧数据
                            info!("Visualizer node: Processing frame input with id 'frame'");
                            
                            // 从元数据中获取图像尺寸
                            let width = match metadata.parameters.get("width") {
                                Some(dora_node_api::Parameter::String(s)) => s.parse::<u32>().ok().unwrap_or(640),
                                Some(dora_node_api::Parameter::Integer(i)) => *i as u32,
                                _ => 640,
                            };
                            let height = match metadata.parameters.get("height") {
                                Some(dora_node_api::Parameter::String(s)) => s.parse::<u32>().ok().unwrap_or(480),
                                Some(dora_node_api::Parameter::Integer(i)) => *i as u32,
                                _ => 480,
                            };
                            
                            info!("Visualizer node: Image dimensions from metadata - {}x{}", width, height);
                            
                            // 获取数据类型和长度
                            let array = data.as_any().downcast_ref::<UInt8Array>()
                                .context("Expected UInt8Array")?;
                            let data_type = array.data_type();
                            let data_length = array.len();
                            info!("Visualizer node: Data type: {:?}", data_type);
                            info!("Visualizer node: Data length: {}", data_length);
                            
                            // 将数据转换为字节向量
                            let img_data: Vec<u8> = array.iter().filter_map(|x| x).collect();
                            info!("Visualizer node: Received frame with {} bytes", img_data.len());
                            
                            // 验证数据长度与元数据中的尺寸是否匹配
                            if img_data.len() != (width * height * 3) as usize {
                                warn!("Visualizer node: Data size mismatch - expected {}, got {}", width * height * 3, img_data.len());
                            }
                            
                            // 创建一个空的Mat，然后使用memcpy将数据复制进去
                            let mut mat = unsafe {
                                Mat::new_rows_cols(height as i32, width as i32, CV_8UC3)?
                            };
                            
                            // 将图像数据复制到Mat中
                            unsafe {
                                // 使用 data_mut 获取可变指针
                                let data_ptr = mat.data_mut() as *mut u8;
                                std::ptr::copy_nonoverlapping(img_data.as_ptr(), data_ptr, img_data.len());
                            }
                            
                            // 在图像上绘制检测框
                            for detection in &last_detections {
                                // 将相对坐标转换为绝对坐标
                                let x = (detection.x * width as f32) as i32;
                                let y = (detection.y * height as f32) as i32;
                                let w = (detection.width * width as f32) as i32;
                                let h = (detection.height * height as f32) as i32;
                                
                                // 确保边界框在图像范围内
                                let x = x.max(0).min(width as i32 - 1);
                                let y = y.max(0).min(height as i32 - 1);
                                let w = w.min(width as i32 - x);
                                let h = h.min(height as i32 - y);
                                
                                // 创建检测框
                                let rect = Rect::new(x, y, w, h);
                                
                                // 获取类别颜色
                                let color = get_class_color(&detection.class_name);
                                
                                // 绘制矩形框
                                imgproc::rectangle(
                                    &mut mat,
                                    rect,
                                    color,
                                    2,  // 线宽
                                    LINE_8,
                                    0,
                                )?;
                                
                                // 添加标签和置信度
                                let label = format!("{}: {:.2}", detection.name, detection.confidence);
                                let class_label = format!("{}: {:.2}%", detection.class_name, detection.confidence * 100.0);
                                
                                // 声明一个变量用于接收基线偏移量
                                let mut baseline = 0;
                                let text_size = imgproc::get_text_size(
                                    &class_label,
                                    FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    1,
                                    &mut baseline,  // 添加第5个参数：基线偏移量的可变引用
                                )?;
                                let bg_rect = Rect::new(
                                    x,
                                    y - text_size.height - 5,
                                    text_size.width + 5,
                                    text_size.height + 5,
                                );
                                imgproc::rectangle(
                                    &mut mat,
                                    bg_rect,
                                    Scalar::new(0.0, 0.0, 0.0, 0.0), // 黑色背景
                                    -1, // 填充矩形
                                    LINE_8,
                                    0,
                                )?;
                                
                                // 绘制类别标签
                                let org = Point::new(x, y - 5);
                                imgproc::put_text(
                                    &mut mat,
                                    &class_label,
                                    org,
                                    FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    Scalar::new(255.0, 255.0, 255.0, 0.0), // 白色文字
                                    1,
                                    LINE_AA,
                                    false,
                                )?;
                                
                                // 绘制对象ID
                                if !detection.name.is_empty() {
                                    let id_org = Point::new(x, y + h + 15);
                                    imgproc::put_text(
                                        &mut mat,
                                        &detection.name,
                                        id_org,
                                        FONT_HERSHEY_SIMPLEX,
                                        0.4,
                                        color,
                                        1,
                                        LINE_AA,
                                        false,
                                    )?;
                                }
                            }
                            
                            // 显示帧计数
                            let counter_text = format!("Frame: {}", frame_counter);
                            imgproc::put_text(
                                &mut mat,
                                &counter_text,
                                Point::new(10, 30),
                                FONT_HERSHEY_SIMPLEX,
                                0.7,
                                Scalar::new(0.0, 255.0, 0.0, 0.0), // 绿色
                                2,
                                LINE_AA,
                                false,
                            )?;
                            
                            // 显示检测数量
                            let detection_text = format!("Objects: {}", last_detections.len());
                            imgproc::put_text(
                                &mut mat,
                                &detection_text,
                                Point::new(10, 60),
                                FONT_HERSHEY_SIMPLEX,
                                0.7,
                                Scalar::new(0.0, 255.0, 0.0, 0.0), // 绿色
                                2,
                                LINE_AA,
                                false,
                            )?;
                            
                            // 显示图像
                            if highgui::imshow("Visualizer - Camera Feed with Detections", &mat).is_ok() {
                                // 检查按键事件 (按q或ESC退出)
                                let key = highgui::wait_key(1).unwrap_or(0);
                                if key == 'q' as i32 || key == 27 { // 'q'键或ESC键退出
                                    info!("Visualizer node: Quit key pressed, stopping...");
                                    break;
                                }
                            } else {
                                warn!("Visualizer node: Failed to display image");
                            }
                            
                            frame_counter += 1;
                            info!("Visualizer node: Frame displayed with {} detections", last_detections.len());
                        }
                        "detections" => {
                            // 处理检测结果
                            info!("Visualizer node: Processing detections input with id 'detections'");
                            
                            // 解析检测结果
                            if let Some(array) = data.as_any().downcast_ref::<UInt8Array>() {
                                let detection_data: Vec<u8> = array.iter().filter_map(|x| x).collect();
                                info!("Visualizer node: Received {} bytes of detection data", detection_data.len());
                                
                                // 解析检测数据
                                // 格式: [name(16字节), class_name(16字节), confidence(4字节), x(4字节), y(4字节), width(4字节), height(4字节)] 重复
                                let detection_size = 16 + 16 + 4 + 4 + 4 + 4 + 4; // 52字节每检测
                                
                                if detection_data.len() % detection_size == 0 {
                                    last_detections.clear();
                                    
                                    for chunk in detection_data.chunks(detection_size) {
                                        if chunk.len() == detection_size {
                                            // 解析name (16字节)
                                            let name_bytes = &chunk[0..16];
                                            let name = str::from_utf8(name_bytes)
                                                .unwrap_or("")
                                                .trim_matches('\0')
                                                .to_string();
                                            
                                            // 解析class_name (16字节)
                                            let class_bytes = &chunk[16..32];
                                            let class_name = str::from_utf8(class_bytes)
                                                .unwrap_or("")
                                                .trim_matches('\0')
                                                .to_string();
                                            
                                            // 解析其他字段
                                            let confidence = f32::from_le_bytes([
                                                chunk[32], chunk[33], chunk[34], chunk[35]
                                            ]);
                                            let x = f32::from_le_bytes([
                                                chunk[36], chunk[37], chunk[38], chunk[39]
                                            ]);
                                            let y = f32::from_le_bytes([
                                                chunk[40], chunk[41], chunk[42], chunk[43]
                                            ]);
                                            let width = f32::from_le_bytes([
                                                chunk[44], chunk[45], chunk[46], chunk[47]
                                            ]);
                                            let height = f32::from_le_bytes([
                                                chunk[48], chunk[49], chunk[50], chunk[51]
                                            ]);
                                            
                                            last_detections.push(Detection {
                                                name,
                                                class_name,
                                                confidence,
                                                x,
                                                y,
                                                width,
                                                height,
                                            });
                                        }
                                    }
                                    info!("Visualizer node: Parsed {} detections", last_detections.len());
                                } else {
                                    error!("Visualizer node: Invalid detection data size: {} (expected multiple of {})", detection_data.len(), detection_size);
                                }
                            }
                        }
                        _ => {
                            info!("Visualizer node: Received input with id '{}', ignoring", id);
                        }
                    }
                }
                Event::Stop(_) => {
                    info!("Visualizer node: Received stop event");
                    break;
                }
                Event::Error(e) => {
                    error!("Visualizer node: Received error event: {}", e);
                    continue; // 不退出，继续运行
                }
                _ => {
                    info!("Visualizer node: Received other event: {:?}", event);
                }
            }
        }
    }
    
    // 销毁窗口
    highgui::destroy_all_windows()?;
    info!("Visualizer node: Finished");
    
    Ok(())
}

