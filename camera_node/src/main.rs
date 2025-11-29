use dora_node_api::{DoraNode, Event, dora_core::config::DataId, MetadataParameters};
use opencv::{
    core::{Mat, Scalar},
    highgui,
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTrait, VideoCaptureTraitConst, CAP_ANY},
};
use std::time::Duration;

fn main() {
    println!("Camera node: Starting...");
    
    // 初始化Dora节点
    let (mut node, mut event_stream) = match DoraNode::init_from_env() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Camera node: Failed to initialize DoraNode: {}", e);
            std::process::exit(1);
        }
    };

    println!("Camera node: Dora node initialized successfully");

    // 初始化摄像头
    println!("Camera node: Attempting to open camera at index 0");
    let mut cam = VideoCapture::new(0, CAP_ANY).unwrap();
    if !cam.is_opened().unwrap() {
        eprintln!("Camera node: Failed to open camera");
        std::process::exit(1);
    }

    // 设置摄像头分辨率
    cam.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0).unwrap();
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0).unwrap();

    // 获取实际分辨率
    let width = cam.get(videoio::CAP_PROP_FRAME_WIDTH).unwrap_or(640.0);
    let height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT).unwrap_or(480.0);
    println!("Camera node: Camera opened successfully - {}x{}", width as i32, height as i32);

    // 预热摄像头
    println!("Camera node: Warming up camera...");
    std::thread::sleep(Duration::from_millis(1000));
    for _ in 0..5 {
        let mut frame = Mat::default();
        if cam.read(&mut frame).unwrap() {
            // 丢弃预热帧
        }
    }
    println!("Camera node: Warmup complete");

    // 初始化OpenCV窗口
    highgui::named_window("Camera Feed", highgui::WINDOW_AUTOSIZE).unwrap();

    let mut frame_count = 0;
    let start_time = std::time::Instant::now();

    // 主循环 - 等待输入事件来触发帧捕获
    loop {
        if let Some(event) = event_stream.recv_timeout(Duration::from_millis(10)) {
            match event {
                Event::Input { id, data: _, metadata: _ } => {
                    if id.as_str() == "tick" {
                        // 读取帧
                        let mut frame = Mat::default();
                        if !cam.read(&mut frame).unwrap() {
                            eprintln!("Camera node: Failed to read frame");
                            continue;
                        }

                        if frame.size().unwrap().width <= 0 || frame.size().unwrap().height <= 0 {
                            eprintln!("Camera node: Empty frame received");
                            continue;
                        }

                        // 在图像上添加文本
                        imgproc::put_text(
                            &mut frame,
                            &format!("Frame: {}", frame_count),
                            opencv::core::Point::new(10, 30),
                            imgproc::FONT_HERSHEY_SIMPLEX,
                            1.0,
                            Scalar::new(255.0, 255.0, 200.0, 0.0), // 白色文本
                            2,
                            imgproc::LINE_AA,
                            false,
                        ).unwrap();

                        // 显示图像
            //            highgui::imshow("Camera Feed", &frame).unwrap();

                        // 将OpenCV Mat转换为字节数组 - BGR格式
                        let size = frame.size().unwrap();
                        let channels = frame.channels();
                        let expected_size = (size.width * size.height * channels) as usize;
                        
                        let mat_data = unsafe {
                            std::slice::from_raw_parts(
                                frame.data(), 
                                expected_size
                            ).to_vec()
                        };
                        
                        // 验证数据大小
                        let actual_width = size.width as i32;
                        let actual_height = size.height as i32;
                        let actual_channels = channels;
                        let calculated_size = (actual_width * actual_height * actual_channels) as usize;
                        
                        println!("Camera node: Frame size: {}, Data length: {}, Calculated: {}x{}x{}={}", 
                                frame_count, mat_data.len(), actual_width, actual_height, actual_channels, calculated_size);

                        // 使用正确的API发送数据
                        let output_id = DataId::from("frame".to_string());
                     //   let parameters = MetadataParameters::default();
                        
                        let mut parameters = MetadataParameters::new();
                        parameters.insert("width".to_string(), dora_node_api::Parameter::String(actual_width.to_string()));
                        parameters.insert("height".to_string(), dora_node_api::Parameter::String(actual_height.to_string()));
                        parameters.insert("channels".to_string(), dora_node_api::Parameter::String(actual_channels.to_string()));
                        parameters.insert("frame_id".to_string(), dora_node_api::Parameter::String(frame_count.to_string()));
                        
                        match node.send_output_bytes(output_id, parameters, mat_data.len(), &mat_data) {
                            Ok(_) => {
                                frame_count += 1;
                                println!("Camera node: Sent frame {}", frame_count);
                            },
                            Err(e) => {
                                eprintln!("Camera node: Failed to send frame: {}", e);
                                // 继续运行，不退出
                            }
                        }
                    }
                }
                Event::Stop(_) => {
                    println!("Camera node: Received stop event after sending {} frames", frame_count);
                    break;
                }
                Event::Error(e) => {
                    println!("Camera node: Received error event: {}", e);
                    continue; // 继续运行
                }
                _ => {
                    println!("Camera node: Received other event: {:?}", event);
                }
            }
        }

        // 检查是否有按键
        let key = highgui::wait_key(1).unwrap_or(0);
        if key == 'q' as i32 || key == 27 { // 'q'键或ESC键退出
            println!("Camera node: Quit key pressed, stopping...");
            break;
        }

        // 控制帧率
        std::thread::sleep(Duration::from_millis(33)); // ~30 FPS
    }

    // 销毁窗口
    highgui::destroy_all_windows().unwrap();

    println!("Camera node: Finished, sent {} frames total", frame_count);
}
