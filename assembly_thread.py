import time
import os
import numpy as np
import mujoco
import mujoco.viewer
import cv2
import threading
from ultralytics import YOLO
import queue

# 优化YOLO模型加载
yolo_model = YOLO('yolo_train/weights/best.pt')
# 预热模型以提高推理速度
dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
_ = yolo_model(dummy_image, verbose=False)

global detected
global detected_depth
global detected_xywh
global filtered_depth
global filtered_xywh
global target_lost_count
global emergency_stop

target_xywh = [320, 240, 280, 280]
detected = False
filtered_depth = 0.5
filtered_xywh = [320, 240, 280, 280]
target_lost_count = 0
emergency_stop = False

# 线程同步
detection_lock = threading.Lock()
control_lock = threading.Lock()

# 控制参数
Kp = 0.3  # 比例增益，降低以提高稳定性
Ki = 0.01  # 积分增益
Kd = 0.05  # 微分增益
max_velocity = 0.5  # 最大速度限制

# 滤波参数
alpha_depth = 0.7  # 深度滤波系数
alpha_xywh = 0.8   # 位置滤波系数

# 鲁棒性参数
target_lost_threshold = 10  # 目标丢失帧数阈值
max_control_error = 0.1     # 最大控制误差阈值
safety_velocity_limit = 0.3  # 安全速度限制

def apply_filtering():
    """应用滤波到检测结果"""
    global filtered_depth, filtered_xywh, detected_depth, detected_xywh
    
    with detection_lock:
        if detected:
            # 深度滤波
            filtered_depth = alpha_depth * filtered_depth + (1 - alpha_depth) * detected_depth
            
            # 位置滤波
            for i in range(4):
                filtered_xywh[i] = alpha_xywh * filtered_xywh[i] + (1 - alpha_xywh) * detected_xywh[i]

# init_camera函数已移除，新渲染器API不需要GLFW窗口

def setup_camera(model, data, camera_name, width=640, height=480):
    """设置摄像头参数 - 使用新的渲染器API"""
    # 获取摄像头ID
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    
    # 增强光照设置 - 修复相机全黑问题
    model.vis.headlight.ambient[:] = [0.8, 0.8, 0.8]  # 大幅增加环境光
    model.vis.headlight.diffuse[:] = [1.0, 1.0, 1.0]  # 最大漫反射光
    model.vis.headlight.specular[:] = [0.5, 0.5, 0.5]  # 增加镜面反射
    
    # 使用新的渲染器API，避免OpenGL上下文问题
    renderer = mujoco.Renderer(model, height=height, width=width)
    
    return camera_id, renderer, camera_name

def get_camera_image(model, data, camera_id, renderer, camera_name):
    """获取摄像头图像 - 使用新的渲染器API"""
    # 使用新渲染器渲染
    renderer.update_scene(data, camera=camera_name)
    rgb_image = renderer.render()
    
    # OpenCV使用BGR格式，需要转换
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    # 获取真实深度图 - 使用MuJoCo的深度渲染功能
    depth_image = renderer.render(depth=True)
    
    # 将深度图转换为米为单位
    # MuJoCo深度图范围通常在[0,1]，需要根据相机参数转换
    mjr_znear = 0.05  # 近平面
    mjr_zfar = 8.0    # 远平面
    
    # 将归一化深度转换为真实深度（米）
    depth = mjr_znear * mjr_zfar / (mjr_zfar - depth_image * (mjr_zfar - mjr_znear))
    
    return bgr, depth

def camera_rendering_thread(model, data):
    """独立线程处理摄像头渲染 - 使用新的渲染器API"""
    # 不再需要GLFW窗口，新渲染器API不需要OpenGL上下文
    camera_id, renderer, camera_name = setup_camera(model, data, "ego_camera")

    mjr_znear = 0.05
    mjr_zfar = 8
    global detected_depth
    global detected_xywh
    global detected

    frame_count = 0
    last_results = None
    
    while True:
        rgb_image, depth_map = get_camera_image(model, data, camera_id, renderer, camera_name)
        
        # 优化：每2帧进行一次YOLO检测以提高性能
        if frame_count % 2 == 0:
            results = yolo_model.predict(rgb_image, verbose=False, conf=0.5)  # 提高置信度阈值
            last_results = results
        else:
            # 使用上一帧的结果
            results = last_results
            
        # Draw bounding boxes and class labels on the frame
        if results is not None:
            frame = results[0].plot()
        else:
            frame = rgb_image.copy()
            
        frame_count += 1

        if results is not None and results[0].boxes:
            detected = True
            target_lost_count = 0  # 重置丢失计数
        else:
            detected = False
            target_lost_count += 1
            
        # 检查是否触发紧急停止
        if target_lost_count > target_lost_threshold:
            emergency_stop = True
            print(f"目标丢失超过{target_lost_threshold}帧，触发紧急停止")
        
        # 应用滤波
        apply_filtering()
        
        # Process detection results and extract bounding boxes
        for result in results[0].boxes:
            
            # Get the coordinates (x, y, width, height) of the bounding box
            x_center, y_center, width, height = result.xywh[0]
            detected_xywh = [x_center.cpu(), y_center.cpu(), width.cpu(), height.cpu()]

            confidence = result.conf[0]
            class_id = result.cls[0]

            # Get class name from results
            class_name = results[0].names[int(class_id)]

            # Print the detected object's class and coordinates
            # print(f"Object: {class_name}, Coordinates: ({x_center}, {y_center}, {width}, {height}), Confidence: {confidence:.2f}")
            # 获取目标中心点的深度值（已经是真实深度，单位：米）
            detected_depth = depth_map[int(y_center), int(x_center)]
            
            # 添加深度有效性检查
            if detected_depth <= 0 or detected_depth > 10:  # 深度范围检查
                detected_depth = 0.5  # 使用默认深度
            # print("Depth:", detected_depth)


            cv2.drawMarker(frame, (int(x_center - width / 2), int(y_center - width / 2)), color=(0, 255, 0), thickness=2)
            cv2.drawMarker(frame, (int(x_center + width / 2), int(y_center - width / 2)), color=(0, 255, 0), thickness=2)
            cv2.drawMarker(frame, (int(x_center - width / 2), int(y_center + width / 2)), color=(0, 255, 0), thickness=2)
            cv2.drawMarker(frame, (int(x_center + width / 2), int(y_center + width / 2)), color=(0, 255, 0), thickness=2)

        cv2.drawMarker(frame, (int(target_xywh[0] - target_xywh[2] / 2), int(target_xywh[1] - target_xywh[2] / 2)), color=(255, 0, 255), thickness=2)
        cv2.drawMarker(frame, (int(target_xywh[0] + target_xywh[2] / 2), int(target_xywh[1] - target_xywh[2] / 2)), color=(255, 0, 255), thickness=2)
        cv2.drawMarker(frame, (int(target_xywh[0] - target_xywh[2] / 2), int(target_xywh[1] + target_xywh[2] / 2)), color=(255, 0, 255), thickness=2)
        cv2.drawMarker(frame, (int(target_xywh[0] + target_xywh[2] / 2), int(target_xywh[1] + target_xywh[2] / 2)), color=(255, 0, 255), thickness=2)
        # 显示状态信息
        status_text = f"Target Lost: {target_lost_count}/{target_lost_threshold}"
        if emergency_stop:
            status_text += " - EMERGENCY STOP"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('MuJoCo Camera', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and emergency_stop:
            # 重置紧急停止状态
            emergency_stop = False
            target_lost_count = 0
            print("紧急停止已重置")
        elif key == ord('e'):
            # 手动触发紧急停止
            emergency_stop = True
            print("手动触发紧急停止")
            
        time.sleep(0.033)  # ~30 FPS

def visual_servo_thread(model, data):

    global detected
    global detected_depth
    global detected_xywh

    def calc_jacobian(model, data):
        jacobian = np.zeros((6, 6))
        p_end = np.array(data.xpos[9])
        R_end = np.array(data.xmat[9]).reshape(3, 3)
        R0_T = np.zeros((6, 6))
        R0_T[:3, :3] = R_end.T
        R0_T[3:, 3:] = R_end.T

        for i in range(6):
            z_i = np.array(data.xaxis[i])
            p_i = np.array(data.xpos[i + 2])

            J_v = np.cross(z_i, p_end - p_i)
            J_w = z_i
            jacobian[:,i] = np.hstack((J_v, J_w)).transpose()
        return R0_T @ jacobian

    def damped_pseudo_inverse(jacobian):  # avoid singularity
        
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        min_sv = np.min(singular_values)
        
        if min_sv < 1e-3:
            lambda_val = (1 - min_sv/1e-3)
        else:
            lambda_val = 1e-6

        jjt = jacobian @ jacobian.T
        damping_matrix = lambda_val**2 * np.eye(jacobian.shape[0])
        j_pinv = jacobian.T @ np.linalg.inv(jjt + damping_matrix)
        return j_pinv
    
    # PID控制器状态
    prev_error = np.zeros(8)
    integral_error = np.zeros(8)
    
    while True:
        # 检查紧急停止状态
        if emergency_stop:
            data.ctrl = [0, 0, 0, 0, 0, 0]
            print("紧急停止激活，机器人已停止")
            time.sleep(0.02)
            continue
            
        if detected:
            # 使用滤波后的数据
            z = filtered_depth
            
            # 更准确的相机内参（基于RealSense D435i典型参数）
            fx = 615.0  # 焦距x
            fy = 615.0  # 焦距y
            cx = 320.0  # 主点x
            cy = 240.0  # 主点y
            
            # 计算四个角点的误差
            corners = []
            target_corners = []
            
            # 四个角点：左上、右上、左下、右下
            corner_offsets = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
            
            for dx, dy in corner_offsets:
                # 当前检测的角点（像素坐标转归一化坐标）
                x = (filtered_xywh[0] + dx * filtered_xywh[2] / 2 - cx) / fx
                y = (filtered_xywh[1] + dy * filtered_xywh[3] / 2 - cy) / fy
                corners.extend([x, y])
                
                # 目标角点（像素坐标转归一化坐标）
                x_d = (target_xywh[0] + dx * target_xywh[2] / 2 - cx) / fx
                y_d = (target_xywh[1] + dy * target_xywh[3] / 2 - cy) / fy
                target_corners.extend([x_d, y_d])
            
            # 计算误差
            e = np.array(corners) - np.array(target_corners)
            
            # 计算图像雅可比矩阵（使用中心点）
            x_center = (filtered_xywh[0] - cx) / fx
            y_center = (filtered_xywh[1] - cy) / fy
            
            # 构建图像雅可比矩阵
            L_e = np.zeros((8, 6))
            for i in range(4):
                idx = i * 2
                x, y = corners[idx], corners[idx + 1]
                L_e[idx:idx+2, :] = np.array([
                    [-1/z, 0, x/z, x*y, -(1+x*x), y],
                    [0, -1/z, y/z, 1+y*y, -x*y, -x]
                ])
            
            # PID控制
            integral_error += e * 0.02  # 积分项
            derivative_error = (e - prev_error) / 0.02  # 微分项
            
            # 计算控制速度
            v_c = Kp * e + Ki * integral_error + Kd * derivative_error
            
            # 检查控制误差是否过大
            error_norm = np.linalg.norm(e)
            if error_norm > max_control_error:
                print(f"控制误差过大: {error_norm:.4f}，停止控制")
                data.ctrl = [0, 0, 0, 0, 0, 0]
                time.sleep(0.02)
                continue
            
            # 计算机器人雅可比矩阵
            J_q = calc_jacobian(model, data)
            
            # 计算关节速度
            try:
                ctrl = -damped_pseudo_inverse(J_q) @ damped_pseudo_inverse(L_e) @ v_c
                
                # 应用安全速度限制
                ctrl = np.clip(ctrl, -safety_velocity_limit, safety_velocity_limit)
                data.ctrl = ctrl.tolist()
                
                # 打印调试信息
                if error_norm > 0.01:  # 只在误差较大时打印
                    print(f"误差范数: {error_norm:.4f}, 深度: {z:.3f}m, 最大速度: {np.max(np.abs(ctrl)):.3f}")
                    
            except np.linalg.LinAlgError:
                print("矩阵求逆失败，停止控制")
                data.ctrl = [0, 0, 0, 0, 0, 0]
            
            prev_error = e.copy()
            
        else:
            data.ctrl = [0, 0, 0, 0, 0, 0]
            # 重置PID状态
            prev_error = np.zeros(8)
            integral_error = np.zeros(8)

        time.sleep(0.02)

def main():
    xml_path = os.path.join(os.path.dirname(__file__), './scene.xml')
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after 30 wall-seconds.

        d.ctrl = [0, 0, 0, 0, 0, 0] #left and right actuator
        camera_thread = threading.Thread(
            target=camera_rendering_thread, 
            args=(m, d)
        )
        camera_thread.daemon = True
        camera_thread.start()

        viservo_thread = threading.Thread(
            target=visual_servo_thread, 
            args=(m, d)
        )
        viservo_thread.daemon = True
        viservo_thread.start()

        while viewer.is_running():#  and time.time() - start < 30
            mujoco.mj_step(m, d)
            viewer.sync()
    
if __name__ == "__main__":
    main()