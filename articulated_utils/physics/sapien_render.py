import numpy as np
import sapien.core as sapien
import cv2
from articulated_utils.utils.utils import (
    join_path,
    save_json,
)
import seaborn as sns
from PIL import ImageColor
from omegaconf import DictConfig
import os
import colorsys


def setup_cameras(cfg, scene):
    cameras = {}
    for view_name, view_params in cfg.camera_params.views.items():
        camera = scene.add_camera(
            name=view_name,
            width=cfg.camera_params.width,
            height=cfg.camera_params.height,
            fovy=np.deg2rad(35),
            near=0.1,
            far=100,
        )
        camera_mount_actor = scene.create_actor_builder().build_kinematic()
        camera.set_parent(parent=camera_mount_actor, keep_pose=False)

        # set the camera up at `cam_pos` and looking at `look_at`
        cam_pos = np.array(view_params.cam_pos)
        look_at = np.array(view_params.look_at)
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_mount_actor.set_pose(
            sapien.Pose.from_transformation_matrix(mat44))

        cameras[view_name] = camera

    return cameras


def flip_video(video_path: str):
    """
    Flip a video file so it plays in reverse.

    Args:
    video_path (str): Path to the video file to be flipped.

    Returns:
    None. The original video file is replaced with the flipped version.
    """
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a temporary file for the flipped video
    temp_path = join_path(os.path.dirname(video_path), "temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Write frames in reverse order
    for frame in reversed(frames):
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

    # Replace the original video with the flipped one
    os.replace(temp_path, video_path)


def setup_video_writers(cfg, joint_name):
    video_writers = {}
    for camera_view in cfg.camera_params.views:
        video_file = join_path(
            cfg.output.dir, f"video_{joint_name}_{camera_view}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writers[camera_view] = cv2.VideoWriter(
            video_file, fourcc, 24.0, (cfg.camera_params.width,
                                       cfg.camera_params.height)
        )
    return video_writers


class VideoWriterManager:
    def __init__(self, cfg: DictConfig, joint_name: str):
        self.cfg = cfg
        self.joint_name = joint_name
        self.video_writers = None

    def __enter__(self):
        self.video_writers = setup_video_writers(self.cfg, self.joint_name)
        return self.video_writers

    def __exit__(self, exc_type, exc_value, traceback):
        if self.video_writers:
            for writer in self.video_writers.values():
                writer.release()


def apply_morphological_operations(label_image):
    """Apply morphological operations to reduce noise in the segmentation mask."""
    kernel = np.ones((5, 5), np.uint8)
    label_image = cv2.morphologyEx(label_image, cv2.MORPH_OPEN, kernel)
    label_image = cv2.morphologyEx(label_image, cv2.MORPH_CLOSE, kernel)
    return label_image


def get_segmentation_data(camera):
    """Get segmentation labels and unique labels from the camera."""
    # Capture a frame
    camera.take_picture()

    # Get the segmentation labels
    seg_labels = camera.get_visual_actor_segmentation()

    # Extract the label image (second channel of seg_labels)
    label_image = seg_labels[..., 1].astype(np.uint8)

    # Apply morphological operations to clean the segmentation mask
    cleaned_label_image = apply_morphological_operations(label_image)

    # Get unique labels from the cleaned label image
    unique_labels = np.unique(cleaned_label_image)

    return cleaned_label_image, unique_labels


def get_distinct_colors(num_colors):
    # return sns.color_palette("hsv", num_colors).as_hex()
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors


def create_color_mappings(unique_labels, robot):
    """Create color mappings for links and labels."""
    # Generate distinct colors for each unique label
    distinct_colors = get_distinct_colors(len(unique_labels))
    color_palette = np.array(
        [ImageColor.getrgb(color) for color in distinct_colors], dtype=np.uint8
    )

    # Create a mapping of link IDs to link names
    link_id_to_name = {int(link.id): link.name for link in robot.get_links()}

    # Create a mapping of label IDs to colors
    link_id_to_color = {
        label: color_palette[i % len(distinct_colors)]
        for i, label in enumerate(unique_labels)
    }

    # Create a mapping of link names to color hex codes
    # Only includes links that are both in the robot model and in the segmentation
    link_color_mapping = {
        link_id_to_name[label]: "#{:02x}{:02x}{:02x}".format(*color)
        for label, color in link_id_to_color.items()
        if label in link_id_to_name
    }

    return link_id_to_name, link_id_to_color, link_color_mapping


def create_segmentation_image(
    label_image, link_id_to_color, link_id_to_name, object_white
):
    """Create a segmentation image based on the label image and color mappings."""
    # Initialize an empty color image
    color_image = np.zeros(
        (label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8
    )

    if object_white:
        # If object_white is True, set all object parts to white
        # because the background will not be a link in the `link_id_to_name` mapping
        for label, color in link_id_to_color.items():
            if (
                label in link_id_to_name
            ):  # This check ensures we're only coloring actual robot parts
                color_image[label_image == label] = [255, 255, 255]
    else:
        # If object_white is False, color each part with its assigned color
        for label, color in link_id_to_color.items():
            color_image[label_image == label] = color

    return color_image


def take_camera_pic(
    robot,
    camera,
    use_segmentation=True,
    output_json="seg.json",
    object_white=True,
):
    """Take a picture from the camera and process it based on the given parameters."""
    camera.take_picture()
    if not use_segmentation:
        # If segmentation is not used, return the regular color image
        rgba = camera.get_float_texture("Color")
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        return cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGR)

    # Get segmentation data
    label_image, unique_labels = get_segmentation_data(camera)

    # Create color mappings
    link_id_to_name, link_id_to_color, link_color_mapping = create_color_mappings(
        unique_labels, robot
    )

    # Create the segmentation image
    color_image = create_segmentation_image(
        label_image, link_id_to_color, link_id_to_name, object_white
    )

    if object_white:
        # If object_white is True, update the color mapping to reflect the binary coloring
        link_color_mapping = {k: "#FFFFFF" for k in link_color_mapping}
        link_color_mapping["background"] = "#000000"

    if use_segmentation:
        # Save the color mapping to a JSON file
        save_json(link_color_mapping, output_json)

    # Convert the color image from RGB to BGR (OpenCV format) and return
    return cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)


def save_image(image, filename):
    cv2.imwrite(filename, image)


class SapienRenderer:
    """简化的SAPIEN渲染器，专用于critic-verifier系统"""
    
    def __init__(self, resolution=[512, 512]):
        self.resolution = resolution
        self.engine = None
        self.scene = None
        
    def _init_engine(self):
        """初始化SAPIEN引擎"""
        try:
            self.engine = sapien.Engine()
            renderer = sapien.SapienRenderer()
            self.engine.set_renderer(renderer)
            
            # 创建场景
            scene_config = sapien.SceneConfig()
            self.scene = self.engine.create_scene(scene_config)
            
            # 添加光照
            self.scene.set_ambient_light([0.5, 0.5, 0.5])
            self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
            
            return True
            
        except Exception as e:
            print(f"初始化SAPIEN引擎失败: {e}")
            return False
    
    def render_urdf(self, urdf_path, output_path, camera_pos=[2, 0, 1], camera_target=[0, 0, 0.5]):
        """
        渲染URDF文件到图像
        
        Args:
            urdf_path: URDF文件路径
            output_path: 输出图像路径
            camera_pos: 相机位置
            camera_target: 相机目标点
            
        Returns:
            bool: 是否成功渲染
        """
        try:
            # 初始化引擎（如果尚未初始化）
            if self.engine is None:
                if not self._init_engine():
                    return False
            
            # 加载URDF - 设置正确的工作目录
            urdf_dir = os.path.dirname(urdf_path)
            urdf_filename = os.path.basename(urdf_path)
            
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            
            # 切换到URDF文件所在目录，这样相对路径的mesh文件就能被找到
            original_cwd = os.getcwd()
            try:
                os.chdir(urdf_dir)
                # 创建机器人
                robot = loader.load(urdf_filename)
            finally:
                os.chdir(original_cwd)
            if robot is None:
                print(f"加载URDF失败: {urdf_path}")
                return False
            
            # 设置相机
            camera = self.scene.add_camera(
                name="render_camera",
                width=self.resolution[0],
                height=self.resolution[1],
                fovy=np.deg2rad(35),
                near=0.1,
                far=100,
            )
            
            # 设置相机位置
            camera_mount = self.scene.create_actor_builder().build_kinematic()
            camera.set_parent(parent=camera_mount, keep_pose=False)
            
            # 计算相机矩阵
            cam_pos = np.array(camera_pos)
            look_at = np.array(camera_target)
            forward = look_at - cam_pos
            forward = forward / np.linalg.norm(forward)
            left = np.cross([0, 0, 1], forward)
            left = left / np.linalg.norm(left)
            up = np.cross(forward, left)
            
            mat44 = np.eye(4)
            mat44[:3, :3] = np.stack([forward, left, up], axis=1)
            mat44[:3, 3] = cam_pos
            camera_mount.set_pose(sapien.Pose.from_transformation_matrix(mat44))
            
            # 渲染
            self.scene.step()
            self.scene.update_render()
            camera.take_picture()
            
            # 获取图像
            rgba = camera.get_float_texture("Color")
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            rgb_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGR)
            
            # 保存图像
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, rgb_img)
            
            # 清理
            self.scene.remove_articulation(robot)
            self.scene.remove_camera(camera)
            self.scene.remove_actor(camera_mount)
            
            return True
            
        except Exception as e:
            print(f"SAPIEN渲染失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """清理资源"""
        if self.scene:
            self.scene = None
        if self.engine:
            self.engine = None
