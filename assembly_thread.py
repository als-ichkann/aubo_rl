import time
import os
import numpy as np
import mujoco
import mujoco.viewer
import cv2
import threading
from ultralytics import YOLO

yolo_model = YOLO('yolo_train/weights/best.pt')

global detected
global detected_depth
global detected_xywh

target_xywh = [320, 240, 280, 280]
detected = False
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
    
    # 创建虚拟深度图（新渲染器API不直接提供深度）
    depth = np.ones((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32) * 0.5
    
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

    while True:
        rgb_image, depth_map = get_camera_image(model, data, camera_id, renderer, camera_name)
        results = yolo_model.predict(rgb_image, verbose=False)
        # Draw bounding boxes and class labels on the frame
        frame = results[0].plot()

        if results[0].boxes:
            detected = True
        else:
            detected = False
        # print(detected)
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
            detected_depth = mjr_znear * mjr_zfar / (mjr_zfar - depth_map[int(y_center), int(x_center)]  * (mjr_zfar - mjr_znear))
            # print("Depth:", detected_depth)


            cv2.drawMarker(frame, (int(x_center - width / 2), int(y_center - width / 2)), color=(0, 255, 0), thickness=2)
            cv2.drawMarker(frame, (int(x_center + width / 2), int(y_center - width / 2)), color=(0, 255, 0), thickness=2)
            cv2.drawMarker(frame, (int(x_center - width / 2), int(y_center + width / 2)), color=(0, 255, 0), thickness=2)
            cv2.drawMarker(frame, (int(x_center + width / 2), int(y_center + width / 2)), color=(0, 255, 0), thickness=2)

        cv2.drawMarker(frame, (int(target_xywh[0] - target_xywh[2] / 2), int(target_xywh[1] - target_xywh[2] / 2)), color=(255, 0, 255), thickness=2)
        cv2.drawMarker(frame, (int(target_xywh[0] + target_xywh[2] / 2), int(target_xywh[1] - target_xywh[2] / 2)), color=(255, 0, 255), thickness=2)
        cv2.drawMarker(frame, (int(target_xywh[0] - target_xywh[2] / 2), int(target_xywh[1] + target_xywh[2] / 2)), color=(255, 0, 255), thickness=2)
        cv2.drawMarker(frame, (int(target_xywh[0] + target_xywh[2] / 2), int(target_xywh[1] + target_xywh[2] / 2)), color=(255, 0, 255), thickness=2)
        cv2.imshow('MuJoCo Camera', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
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
    
    while True:
        #J_q = calc_jacobian(model, data)
        #data.ctrl = damped_pseudo_inverse(J_q) @ np.array([-0.03, 0.03, -0.03, 0, 0, 0])
        if detected:

            z = detected_depth
            f = 400 # focal length in Pixels

            # up left corner point 
            x, y = (detected_xywh[0] - detected_xywh[2] / 2 - 320) / f, (240 - (detected_xywh[1] - detected_xywh[2] / 2)) / f
            x_d, y_d = (target_xywh[0] - target_xywh[2] / 2 - 320) / f, (240 - (target_xywh[1] - target_xywh[2] / 2)) / f
            e1 = [x - x_d, y - y_d]
            L_e1 = np.array([[-1 / z, 0, x / z, x * y, -(1 + x * x), y], [0, -1 / z, y / z, 1 + y * y, -x * y, -x]])
            # up right corner point 
            x, y = (detected_xywh[0] + detected_xywh[2] / 2 - 320) / f, (240 - (detected_xywh[1] - detected_xywh[2] / 2)) / f
            x_d, y_d = (target_xywh[0] + target_xywh[2] / 2 - 320) / f, (240 - (target_xywh[1] - target_xywh[2] / 2)) / f
            e2 = [x - x_d, y - y_d]
            L_e2 = np.array([[-1 / z, 0, x / z, x * y, -(1 + x * x), y], [0, -1 / z, y / z, 1 + y * y, -x * y, -x]])
            # bottom left corner point 
            x, y = (detected_xywh[0] - detected_xywh[2] / 2 - 320) / f, (240 - (detected_xywh[1] + detected_xywh[2] / 2)) / f
            x_d, y_d = (target_xywh[0] - target_xywh[2] / 2 - 320) / f, (240 - (target_xywh[1] + target_xywh[2] / 2)) / f
            e3 = [x - x_d, y - y_d]
            L_e3 = np.array([[-1 / z, 0, x / z, x * y, -(1 + x * x), y], [0, -1 / z, y / z, 1 + y * y, -x * y, -x]])
            # bottom left corner point 
            x, y = (detected_xywh[0] + detected_xywh[2] / 2 - 320) / f, (240 - (detected_xywh[1] + detected_xywh[2] / 2)) / f
            x_d, y_d = (target_xywh[0] + target_xywh[2] / 2 - 320) / f, (240 - (target_xywh[1] + target_xywh[2] / 2)) / f
            e4 = [x - x_d, y - y_d]
            L_e4 = np.array([[-1 / z, 0, x / z, x * y, -(1 + x * x), y], [0, -1 / z, y / z, 1 + y * y, -x * y, -x]])

            e = np.hstack((e1, e2, e3, e4))
            L_e = np.vstack((L_e1, L_e2, L_e3, L_e4))
            #print("error:", L_e)
            #print("v_c1:", damped_pseudo_inverse(L_e1) @ e1)
            
            J_q = calc_jacobian(model, data)
            ctrl = -0.6 * damped_pseudo_inverse(J_q) @ damped_pseudo_inverse(L_e) @ e
            ctrl = np.clip(ctrl, -1, 1)
            data.ctrl = ctrl.tolist()
            #print("ctrl:", ctrl)
        else:
            data.ctrl = [0, 0, 0, 0, 0, 0]

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