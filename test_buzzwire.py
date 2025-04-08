import numpy as np
import argparse
from gym_dcmm.envs.WireWalkerVecEnv import WireWalkerVecEnv

def test_rewards_simple():
    """
    运行环境并在控制台打印奖励信息
    """
    # 创建环境 - 启用奖励打印
    env = WireWalkerVecEnv(
        task="Tracking",
        track_name="track_0",
        viewer=True,                # 显示 Mujoco 视图器
        print_reward=True,          # 启用详细奖励打印
        imshow_cam=False,
        env_time=30.0,              # 增加模拟时间
        steps_per_policy=20
    )
    
    # 重置环境
    obs, info = env.reset()
    
    step_count = 0
    total_reward = 0
    
    print("\n===== 开始测试 Wire Walker 奖励 =====")
    print("提示: 按空格键暂停/继续模拟\n")
    
    try:
        while True:
            step_count += 1
            
            # 简单测试策略 - 向导线移动并保持一定距离
            wire_pos = obs["wire"]["pos3d"]
            ee_pos = np.concatenate([obs["arm"]["ee_pos3d"][:2], [wire_pos[2]]])
            
            # 计算导线与末端执行器的方向向量
            direction = wire_pos - ee_pos
            distance = np.linalg.norm(direction)
            
            # 基座移动控制 - 向导线方向移动，但速度与距离成正比
            base_action = direction[:2] * min(0.5, distance) 
            base_action = np.clip(base_action, -1, 1)
            
            # 机械臂控制 - 调整高度以接近导线
            arm_action = np.array([0, 0, direction[2] * 0.5, 0]) * 0.01
            arm_action = np.clip(arm_action, -0.025, 0.025)
            
            # 组合动作
            action = {
                "base": base_action,
                "arm": arm_action
            }
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 累计总奖励
            total_reward += reward
            
            # 每10步显示一次累计奖励摘要
            if step_count % 10 == 0:
                print(f"\n步骤 {step_count}:")
                print(f"累计奖励: {total_reward:.4f}")
                print(f"平均奖励: {total_reward/step_count:.4f}")
                print(f"环端距离: {info['ee_distance']:.4f}m")
                print("-" * 30)
            
            if terminated or truncated:
                print(f"\n模拟结束于步骤 {step_count}")
                print(f"最终累计奖励: {total_reward:.4f}")
                print(f"最终平均奖励: {total_reward/step_count:.4f}")
                break
            
    except KeyboardInterrupt:
        print("\n用户中断测试")
    
    finally:
        # 关闭环境
        env.close()
        print("\n===== 测试结束 =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="简单的 Wire Walker 奖励测试")
    args = parser.parse_args()
    
    test_rewards_simple()