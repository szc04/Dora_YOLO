use dora_node_api::{DoraNode, Event, IntoArrow};
use dora_node_api::dora_core::config::DataId;

use opencv::{
    core::Mat,
    prelude::*,
    videoio::{VideoCapture, VideoCaptureTrait, CAP_ANY},
};

fn main() {
    // 延迟一点，确保系统稳定
    std::thread::sleep(std::time::Duration::from_millis(1000));
    eprintln!("Camera  initialize sleep");
    let (mut node, mut event_stream) = match DoraNode::init_from_env() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Camera node: Failed to initialize DoraNode: {}", e);
            std::process::exit(1);
        }
    };
    
    // 尝试创建摄像头
    let mut cam = VideoCapture::default().unwrap();
    
    // 尝试打开摄像头
    if let Err(e) = cam.open(0, CAP_ANY) {
        eprintln!("Camera node: Could not open camera: {}", e);
        std::process::exit(1);
    }

    // 检查摄像头是否打开
    if !cam.is_opened().unwrap_or(false) {
        eprintln!("Camera node: Camera is not opened");
        std::process::exit(1);
    }

    let mut frame = Mat::default();
    let mut frame_count = 0;
    
    loop {
        if let Some(event) = event_stream.recv_timeout(std::time::Duration::from_millis(10)) {
            if let Event::Stop(_) = event {
                eprintln!("Camera node: Received stop event");
                break;
            }
        }

        match cam.read(&mut frame) {
            Ok(success) => {
                if success && !frame.empty() {
                    match frame.data_typed::<u8>() {
                        Ok(data) => {
                            let data = data.to_vec();
                            match node.send_output(DataId::from("frame".to_string()), Default::default(), data.into_arrow()) {
                                Ok(_) => {
                                    frame_count += 1;
                                    if frame_count % 30 == 0 {
                                        println!("Camera node: Sent {} frames", frame_count);
                                    }
                                },
                                Err(e) => {
                                    eprintln!("Camera node: Failed to send output: {}", e);
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Camera node: Failed to get frame data: {}", e);
                            continue;
                        }
                    }
                } else {
                    eprintln!("Camera node: Frame is empty or read failed");
                }
            }
            Err(e) => {
                eprintln!("Camera node: Failed to read frame: {}", e);
                continue;
            }
        }
        
        std::thread::sleep(std::time::Duration::from_millis(33));
    }
    
    eprintln!("Camera node: Finished");
}

