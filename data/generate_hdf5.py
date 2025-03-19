import h5py
import numpy as np
import pandas as pd
import os
import pyarrow.parquet as pq
import imageio


def create_hdf5_file(output_path, episode_data):
    """
    创建符合LeRobot数据规范的HDF5文件
    
    Args:
        output_path (str): 输出HDF5文件路径
        episode_data (dict): 包含以下字段的episode数据:
            - qpos (np.ndarray): [steps, 28] 机器人状态数据
            - action (np.ndarray): [steps, 28] 动作数据
            - images (dict): 包含四个摄像头的视频帧数据:
                - cam_left_high (np.ndarray): [steps, 3, 480, 640]
                - cam_right_high (np.ndarray): [steps, 3, 480, 640]
                - cam_left_wrist (np.ndarray): [steps, 3, 480, 640]
                - cam_right_wrist (np.ndarray): [steps, 3, 480, 640]
    Returns:
        None
    """
    with h5py.File(output_path, 'w') as f:
        # 创建根级元数据
        f.attrs['codebase_version'] = 'v2.0'
        f.attrs['robot_type'] = 'Unitree_G1'
        
        # 创建observation组
        obs_group = f.create_group('observation')
        
        # 创建state数据集
        state_ds = obs_group.create_dataset(
            'qpos', 
            data=episode_data['qpos'],
            dtype=np.float32,
            compression="gzip",
            compression_opts=3
        )
        state_ds.attrs['names'] = [
            "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw",
            "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
            "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
            "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
            "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
            "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0",
            "kLeftHandIndex1", "kRightHandThumb0", "kRightHandThumb1",
            "kRightHandThumb2", "kRightHandIndex0", "kRightHandIndex1",
            "kRightHandMiddle0", "kRightHandMiddle1"
        ]
        
        # 创建images子组
        images_group = obs_group.create_group('images')
        
        # 创建摄像头数据集（以cam_left_high为例，其他同理）
        for cam_name in ['cam_right_high', 
                        'cam_left_wrist', 'cam_right_wrist']:
            cam_ds = images_group.create_dataset(
                cam_name,
                data=episode_data['images'][cam_name],
                dtype=np.uint8,
                chunks=(1,3,480,640),
                compression="lzf"
            )
            cam_ds.attrs['video.fps'] = 30.0
            cam_ds.attrs['video.height'] = 480
            cam_ds.attrs['video.width'] = 640
            cam_ds.attrs['video.channels'] = 3
            cam_ds.attrs['video.codec'] = 'av1'
            cam_ds.attrs['video.pix_fmt'] = 'yuv420p'
            cam_ds.attrs['video.is_depth_map'] = False
            cam_ds.attrs['has_audio'] = False
        
        # 创建动作数据集
        action_ds = f.create_dataset(
            'action',
            data=episode_data['action'],
            dtype=np.float32,
            compression="gzip"
        )
        action_ds.attrs['names'] = state_ds.attrs['names']  # 动作与状态字段一致
        



def read_video(video_path):
    """使用 imageio 读取视频并转换为 numpy 数组"""
    try:
        # 使用 imageio 的 FFmpeg 后端读取视频
        reader = imageio.get_reader(video_path, 'ffmpeg')
        
        # 获取视频元数据
        meta = reader.get_meta_data()
        #assert meta['format'].lower() == 'mp4', f"不支持的视频格式：{meta['format']}"
        #assert meta['source_size'] == (640, 480), f"分辨率不匹配：{meta['source_size']}"

        # 读取所有帧并转换为 CHW 格式
        frames = []
        for frame in reader:
            # 转换为 CHW 格式（HWC → CHW）
            frame = frame.transpose(2, 0, 1)
            frames.append(frame)
        
        # 转换为 numpy 数组并验证形状
        frames_array = np.stack(frames, axis=0).astype(np.uint8)
        assert frames_array.shape[1:] == (3, 480, 640), f"帧形状不匹配：{frames_array.shape}"
        
        return frames_array

    except Exception as e:
        print(f"致命错误：读取视频 {video_path} 失败，错误：{str(e)}")
        return np.empty((0, 3, 480, 640))


def process_episode(parquet_path, video_root_dir,base_output_dir):
    """处理单个episode数据生成HDF5文件"""
    # 解析文件名
    episode_name = os.path.splitext(os.path.basename(parquet_path))[0]
    parquet_file = pq.ParquetFile(parquet_path)
    
    table = parquet_file.read()
    names = table.column_names
    
    # 定义需要的列名
    state_col_name = 'observation.state'  # 根据实际列名修改
    action_col_name = 'action'
    
    # 获取列索引（添加错误处理）
    try:
        state_idx = names.index(state_col_name)
        action_idx = names.index(action_col_name)
    except ValueError as e:
        raise KeyError(f"Parquet文件缺少必要列：{e}, 可用列：{names}")
    
    # 1. 读取原始数据（可能是对象数组）
    raw_state = table.columns[state_idx].to_numpy()
    raw_action = table.columns[action_idx].to_numpy()
    
    # 2. 将嵌套数组转换为二维numpy数组
    state_array = np.array([np.array(item) for item in raw_state])
    action_array = np.array([np.array(item) for item in raw_action])
    
    # 3. 验证数据维度
    assert state_array.ndim == 2, "状态数据必须是二维数组"
    assert action_array.ndim == 2, "动作数据必须是二维数组"
    
    # 4. 验证列数是否符合要求（假设需要28列）
    assert state_array.shape[1] == 28, "状态数据列数不匹配"
    assert action_array.shape[1] == 28, "动作数据列数不匹配"
    
    # 5. 类型转换（如果需要）
    state_array = state_array.astype(np.float32)
    action_array = action_array.astype(np.float32)
    
    # 构建数据字典
    episode_data = {
        'qpos': state_array,
        'action': action_array,
        'images': {}
    }    
    
    # 生成HDF5文件路径
    episode_output_dir = os.path.join(base_output_dir, episode_name)
    os.makedirs(episode_output_dir, exist_ok=True)
    output_path = os.path.join(episode_output_dir, f"{episode_name}.hdf5")

    # 新增：检查文件是否存在
    if os.path.exists(output_path):
        print(f"文件 {output_path} 已存在，跳过处理。")
        return  # 直接返回，跳过后续操作

    right_video_path = os.path.join(
        video_root_dir, 
        'observation.images.cam_right_high', 
        f"{episode_name}.mp4"
    )
    right_video_data = read_video(right_video_path)
    
    # 2. 如果右侧有效，直接使用
    if right_video_data.shape[0] > 0:
        episode_data['images']['cam_right_high'] = right_video_data
    else:
        # 3. 右侧无效，尝试读取左侧cam_left_high
        left_video_path = os.path.join(
            video_root_dir, 
            'observation.images.cam_left_high', 
            f"{episode_name}.mp4"
        )
        left_video_data = read_video(left_video_path)
        if left_video_data.shape[0] > 0:
            episode_data['images']['cam_right_high'] = left_video_data  # 将左侧数据存入右侧键
            print(f"警告：cam_right_high视频读取失败，使用左侧cam_left_high数据替代。")
        else:
            # 4. 双方均失败，记录错误
            print(f"致命错误：cam_right_high和cam_left_high视频均读取失败！")
            return  # 跳过生成HDF5文件

    # 5. 处理其他摄像头（cam_left_wrist和cam_right_wrist）
    for cam_name in ['cam_left_wrist', 'cam_right_wrist']:
        full_cam_dir = f"observation.images.{cam_name}"
        video_path = os.path.join(
            video_root_dir, 
            full_cam_dir, 
            f"{episode_name}.mp4"
        )
        episode_data['images'][cam_name] = read_video(video_path)

    # 验证数据维度一致性
    steps = episode_data['qpos'].shape[0]
    for cam_name in episode_data['images']:
        assert episode_data['images'][cam_name].shape[0] == steps, \
            f"Video {cam_name} has mismatched steps ({steps} vs {episode_data['images'][cam_name].shape[0]})"

    # 生成HDF5文件路径（关键修改部分）
    
    episode_output_dir = os.path.join(base_output_dir, episode_name)
    
    # 创建episode专属目录（如果不存在）
    os.makedirs(episode_output_dir, exist_ok=True)
    
    output_path = os.path.join(
        episode_output_dir,  # 目标目录
        f"{episode_name}.hdf5"  # 文件名
    )
    
    create_hdf5_file(output_path, episode_data)

def main():
    """批量处理所有episode数据"""

    data_dir = "/media/hsh/data/G1_ObjectPlacement_Dataset/data/chunk-000"
    video_root_dir = "/media/hsh/data/G1_ObjectPlacement_Dataset/videos/chunk-000"
    base_output_dir = "/media/hsh/data/G1_ObjectPlacement_Dataset/outputs"
    for parquet_file in os.listdir(data_dir):
        if parquet_file.endswith('.parquet'):
            parquet_path = os.path.join(data_dir, parquet_file)
            process_episode(parquet_path, video_root_dir,base_output_dir)

if __name__ == "__main__":
    main()


