import os
from typing import Any

import imageio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import Coordinate, MultiGrid
from mesa.time import RandomActivation

from agent import Human, Wall, FireExit, Furniture, Fire, Door


class FireEvacuation(Model):
    MIN_HEALTH = 0.75       # 最小健康值
    MAX_HEALTH = 1          # 最大健康值

    MIN_SPEED = 1           # 最小移动速度
    MAX_SPEED = 2           # 最大移动速度

    MIN_NERVOUSNESS = 1     # 最小紧张度
    MAX_NERVOUSNESS = 10    # 最大紧张度

    MIN_EXPERIENCE = 1      # 最小经验值
    MAX_EXPERIENCE = 10     # 最大经验值

    MIN_VISION = 1          # 最小视野范围
    # MAX_VISION is simply the size of the grid

    def __init__(self, floor_plan_file: str, human_count: int, collaboration_percentage: float, fire_probability: float,
                 visualise_vision: bool, random_spawn: bool, save_plots: bool, save_gif: bool, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # 加载楼层平面图
        with open(os.path.join("./floorplans/", floor_plan_file), "rt") as f:
            floorplan = np.matrix([line.strip().split() for line in f.readlines()])

        # 旋转平面图使其与文本文件中的方向一致
        floorplan = np.rot90(floorplan, 3)

        # 获取平面图尺寸
        width, height = np.shape(floorplan)

        # 初始化参数
        self.width = width
        self.height = height
        self.human_count = human_count
        self.collaboration_percentage = collaboration_percentage
        self.visualise_vision = visualise_vision
        self.fire_probability = fire_probability
        self.fire_started = False  # 火灾是否已经发生
        self.save_plots = save_plots

        # 初始化模型组件
        self.schedule = RandomActivation(self)  # 随机激活调度器

        self.grid = MultiGrid(height, width, torus=False)   # 网格环境

        # 用于存储家具位置（可能发生火灾）
        self.furniture: dict[Coordinate, Furniture] = {}

        # 快速查询出口和门的位置
        self.fire_exits: dict[Coordinate, FireExit] = {}
        self.doors: dict[Coordinate, Door] = {}

        # 人员生成位置列表
        self.random_spawn = random_spawn
        self.spawn_pos_list: list[Coordinate] = []

        # 记录帧
        self.frames = []

        self.save_gif_flag = save_gif

        # 解析平面图对象
        for (x, y), value in np.ndenumerate(floorplan):
            pos: Coordinate = (x, y)

            # 根据符号创建对应对象
            value = str(value)
            floor_object = None
            if value == "W":    # 墙壁
                floor_object = Wall(pos, self)
            elif value == "E":  # 安全出口
                floor_object = FireExit(pos, self)
                self.fire_exits[pos] = floor_object
                self.doors[pos] = floor_object  # 出口也是门的一种
            elif value == "F":  # 家具（可能起火）
                floor_object = Furniture(pos, self)
                self.furniture[pos] = floor_object
            elif value == "D":  # 普通门
                floor_object = Door(pos, self)
                self.doors[pos] = floor_object
            elif value == "S":  # 人员生成点
                self.spawn_pos_list.append(pos)

            if floor_object:
                self.grid.place_agent(floor_object, pos)
                self.schedule.add(floor_object)

        # 构建可通行路径图（用于人员路径规划）
        self.graph = nx.Graph()
        for agents, x, y in self.grid.coord_iter():
            pos = (x, y)

            # 如果该位置可通行（无不可通行agent）
            if len(agents) == 0 or not any(not agent.traversable for agent in agents):
                # 添加与相邻位置的连接
                neighbors_pos = self.grid.get_neighborhood(
                    pos, moore=True, include_center=True, radius=1
                )

                for neighbor_pos in neighbors_pos:
                    # If the neighbour position is empty, or no non-traversable contents, add an edge
                    if self.grid.is_cell_empty(neighbor_pos) or not any(
                        not agent.traversable
                        for agent in self.grid.get_cell_list_contents(neighbor_pos)
                    ):
                        self.graph.add_edge(pos, neighbor_pos)

        # 设置数据收集器（用于统计）
        self.datacollector = DataCollector(
            {
                "Alive": lambda m: self.count_human_status(m, Human.Status.ALIVE),
                "Dead": lambda m: self.count_human_status(m, Human.Status.DEAD),
                "Escaped": lambda m: self.count_human_status(m, Human.Status.ESCAPED),
                "Incapacitated": lambda m: self.count_human_mobility(
                    m, Human.Mobility.INCAPACITATED
                ),
                "Normal": lambda m: self.count_human_mobility(m, Human.Mobility.NORMAL),
                "Panic": lambda m: self.count_human_mobility(m, Human.Mobility.PANIC),
                "Verbal Collaboration": lambda m: self.count_human_collaboration(
                    m, Human.Action.VERBAL_SUPPORT
                ),
                "Physical Collaboration": lambda m: self.count_human_collaboration(
                    m, Human.Action.PHYSICAL_SUPPORT
                ),
                "Morale Collaboration": lambda m: self.count_human_collaboration(
                    m, Human.Action.MORALE_SUPPORT
                ),
            }
        )

        # 生成人员agent
        number_collaborators = int(round(self.human_count * (self.collaboration_percentage / 100)))

        for i in range(0, self.human_count):
            # 选择生成位置
            if self.random_spawn:
                pos = self.grid.find_empty()
            else:  # Place human agents at specified spawn locations
                pos = np.random.choice(self.spawn_pos_list)

            if pos:
                # 随机生成人员属性
                health = np.random.randint(self.MIN_HEALTH * 100, self.MAX_HEALTH * 100) / 100
                speed = np.random.randint(self.MIN_SPEED, self.MAX_SPEED)

                # 确定是否具有协作能力
                if number_collaborators > 0:
                    collaborates = True
                    number_collaborators -= 1
                else:
                    collaborates = False

                # 视野分布（基于WHO统计数据）
                vision_distribution = [0.0058, 0.0365, 0.0424, 0.9153]
                vision = int(
                    np.random.choice(
                        np.arange(
                            self.MIN_VISION,
                            self.width + 1,
                            (self.width / len(vision_distribution)),
                        ),
                        p=vision_distribution,
                    )
                )

                # 紧张度分布（中等紧张度概率较高）
                nervousness_distribution = [
                    0.025,
                    0.025,
                    0.1,
                    0.1,
                    0.1,
                    0.3,
                    0.2,
                    0.1,
                    0.025,
                    0.025,
                ]
                nervousness = int(
                    np.random.choice(
                        range(self.MIN_NERVOUSNESS, self.MAX_NERVOUSNESS + 1),
                        p=nervousness_distribution,
                    )
                )

                # 其他属性
                experience = np.random.randint(self.MIN_EXPERIENCE, self.MAX_EXPERIENCE)

                belief_distribution = [0.9, 0.1]  # [Believes, Doesn't Believe]
                believes_alarm = np.random.choice([True, False], p=belief_distribution)

                # 创建人员agent
                human = Human(
                    pos,
                    health=health,
                    speed=speed,
                    vision=vision,
                    collaborates=collaborates,
                    nervousness=nervousness,
                    experience=experience,
                    believes_alarm=believes_alarm,
                    model=self,
                )

                self.grid.place_agent(human, pos)
                self.schedule.add(human)
            else:
                print("No tile empty for human placement!")

        self.running = True

    # Plots line charts of various statistics from a run
    def save_figures(self):
        """保存模拟结果统计图标"""
        DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        OUTPUT_DIR = DIR + "/output"

        results = self.datacollector.get_model_vars_dataframe()

        dpi = 100
        fig, axes = plt.subplots(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi, nrows=1, ncols=3)

        status_results = results.loc[:, ["Alive", "Dead", "Escaped"]]
        status_plot = status_results.plot(ax=axes[0])
        status_plot.set_title("Human Status")
        status_plot.set_xlabel("Simulation Step")
        status_plot.set_ylabel("Count")

        mobility_results = results.loc[:, ["Incapacitated", "Normal", "Panic"]]
        mobility_plot = mobility_results.plot(ax=axes[1])
        mobility_plot.set_title("Human Mobility")
        mobility_plot.set_xlabel("Simulation Step")
        mobility_plot.set_ylabel("Count")

        collaboration_results = results.loc[
            :, ["Verbal Collaboration", "Physical Collaboration", "Morale Collaboration"]
        ]
        collaboration_plot = collaboration_results.plot(ax=axes[2])
        collaboration_plot.set_title("Human Collaboration")
        collaboration_plot.set_xlabel("Simulation Step")
        collaboration_plot.set_ylabel("Successful Attempts")
        collaboration_plot.set_ylim(ymin=0)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt.suptitle(
            "Percentage Collaborating: "
            + str(self.collaboration_percentage)
            + "%, Number of Human Agents: "
            + str(self.human_count),
            fontsize=16,
        )
        save_path = os.path.join(OUTPUT_DIR, "model_graphs", timestr + ".png")

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)

    def render_grid(self):
        """将当前网格绘制为图像帧（RGB）"""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.grid.width)
        ax.set_ylim(0, self.grid.height)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        color_map = {
            Wall: "black",
            FireExit: "green",
            Furniture: "saddlebrown",
            Door: "blue",
            Fire: "red",
            Human: "orange"
        }

        for (content, x, y) in self.grid.coord_iter():
            for agent in content:
                agent_type = type(agent)
                color = color_map.get(agent_type, "gray")
                ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color))

        plt.tight_layout()
        fig.canvas.draw()

        # 获取 RGBA 图像并转换为 RGB numpy 数组
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[..., :3]  # 只保留 RGB，不要 Alpha
        plt.close(fig)
        return image

    def save_gif(self):
        """将记录的图像帧保存为动图"""
        timestr = time.strftime("%Y%m%d-%H%M%S")
        DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        OUTPUT_DIR = os.path.join(DIR, "output", "animations")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"evacuation_{timestr}.gif")

        imageio.mimsave(path, self.frames, fps=2)
        print(f"Simulation saved as GIF to {path}")

    # Starts a fire at a random piece of furniture with file_probability chance
    def start_fire(self):
        """随机选择一个家具位置开始火灾"""
        rand = np.random.random()
        if rand < self.fire_probability:
            fire_furniture: Furniture = np.random.choice(list(self.furniture.values()))
            pos = fire_furniture.pos

            fire = Fire(pos, self)
            self.grid.place_agent(fire, pos)
            self.schedule.add(fire)

            self.fire_started = True
            print(f"Fire started at position {pos}")

    def step(self):
        """
        执行一个模拟步长
        """

        self.schedule.step()

        # If there's no fire yet, attempt to start one
        if not self.fire_started:
            self.start_fire()

        # 记录当前帧
        frame = self.render_grid()
        self.frames.append(frame)

        self.datacollector.collect(self)

        # 如果没有人员存活或所有人员均已疏散，则停止
        if self.count_human_status(self, Human.Status.ALIVE) == 0:
            self.running = False

            if self.save_plots:
                self.save_figures()
            if self.save_gif_flag:
                self.save_gif()

    @staticmethod
    def count_human_collaboration(model, collaboration_type):
        """
        统计特定类型的协作行为次数
        """

        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human):
                if collaboration_type == Human.Action.VERBAL_SUPPORT:
                    count += agent.get_verbal_collaboration_count()
                elif collaboration_type == Human.Action.MORALE_SUPPORT:
                    count += agent.get_morale_collaboration_count()
                elif collaboration_type == Human.Action.PHYSICAL_SUPPORT:
                    count += agent.get_physical_collaboration_count()

        return count

    @staticmethod
    def count_human_status(model, status):
        """
        统计特定状态的人员数量
        """
        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human) and agent.get_status() == status:
                count += 1

        return count

    @staticmethod
    def count_human_mobility(model, mobility):
        """
        统计特定移动能力的人员数量
        """
        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human) and agent.get_mobility() == mobility:
                count += 1

        return count
