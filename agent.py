from typing import Union
from typing_extensions import Self
from mesa.space import Coordinate
import networkx as nx
import numpy as np
from enum import IntEnum
from mesa import Agent
from copy import deepcopy

from utils import get_random_id


def get_line(start, end):
    """
    实现 Bresenham 线算法
    返回从起始坐标到结束坐标的所有像素点坐标（包含起点和终点）
    """
    # 拆解起点和终点的坐标
    x1, y1 = start
    x2, y2 = end

    # 计算 x 和 y 的差值
    diff_x = x2 - x1
    diff_y = y2 - y1

    # 判断线是否陡峭（即 y 的变化比 x 更大）
    line_is_steep = abs(diff_y) > abs(diff_x)

    # 如果是陡峭的线，进行坐标轴变换（交换 x 和 y）
    if line_is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # 如果起点在终点的右边，交换起点和终点
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # 再次计算差值（可能已交换过坐标）
    diff_x = x2 - x1
    diff_y = y2 - y1

    # 设置误差项，用于控制 y 的更新
    error_margin = int(diff_x / 2.0)
    # 确定 y 的步进方向（向上或向下）
    step_y = 1 if y1 < y2 else -1

    y = y1
    path = []  # 用于保存线上的所有点

    # 遍历 x 坐标范围内的每个点，逐步生成路径点
    for x in range(x1, x2 + 1):
        # 根据是否陡峭来决定坐标顺序
        coord = (y, x) if line_is_steep else (x, y)
        path.append(coord)  # 将该点加入路径

        # 更新误差项
        error_margin -= abs(diff_y)

        # 如果误差项小于 0，说明需要调整 y 值
        if error_margin < 0:
            y += step_y
            error_margin += diff_x

    # 如果之前交换了起点和终点，说明路径是反的，需要反转回来
    if swapped:
        path.reverse()

    return path  # 返回最终的路径点列表


"""
FLOOR STUFF
"""


class FloorObject(Agent):
    def __init__(
            self,
            pos: Coordinate,
            traversable: bool,
            flammable: bool,
            spreads_smoke: bool,
            visibility: int = 2,
            model=None,
    ):
        rand_id = get_random_id()
        super().__init__(rand_id, model)
        self.pos = pos
        self.traversable = traversable
        self.flammable = flammable
        self.spreads_smoke = spreads_smoke
        self.visibility = visibility

    def get_position(self):
        return self.pos


class Sight(FloorObject):
    def __init__(self, pos, model):
        super().__init__(
            pos, traversable=True, flammable=False, spreads_smoke=True, visibility=-1, model=model
        )

    def get_position(self):
        return self.pos


class Door(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=True, flammable=False, spreads_smoke=True, model=model)


class FireExit(FloorObject):
    def __init__(self, pos, model):
        super().__init__(
            pos, traversable=True, flammable=False, spreads_smoke=False, visibility=6, model=model
        )


class Wall(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=False, flammable=False, spreads_smoke=False, model=model)


class Furniture(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=False, flammable=True, spreads_smoke=True, model=model)


"""
FIRE STUFF
"""


class Fire(FloorObject):
    """
    A fire agent

    Attributes:
        ...
    """

    def __init__(self, pos, model):
        super().__init__(
            pos,
            traversable=False,
            flammable=False,
            spreads_smoke=True,
            visibility=20,
            model=model,
        )
        self.smoke_radius = 1

    def step(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False, radius=self.smoke_radius
        )

        for neighbor_pos in neighborhood:
            contents = self.model.grid.get_cell_list_contents(neighbor_pos)

            if len(contents) > 0:
                has_smoke = False
                has_fire = False
                for agent in contents:
                    if isinstance(agent, Smoke):
                        has_smoke = True
                    elif isinstance(agent, Fire):
                        has_fire = True
                    if has_smoke and has_fire:
                        break

                if not has_fire:
                    for agent in contents:
                        if agent.flammable:
                            fire = Fire(neighbor_pos, self.model)
                            self.model.schedule.add(fire)
                            self.model.grid.place_agent(fire, neighbor_pos)
                            break

                if not has_smoke:
                    for agent in contents:
                        if agent.spreads_smoke:
                            smoke = Smoke(neighbor_pos, self.model)
                            self.model.schedule.add(smoke)
                            self.model.grid.place_agent(smoke, neighbor_pos)
                            break

    def get_position(self):
        return self.pos


class Smoke(FloorObject):
    """
    A smoke agent

    Attributes:
        ...
    """

    def __init__(self, pos, model):
        super().__init__(pos, traversable=True, flammable=False, spreads_smoke=False, model=model)
        self.smoke_radius = 1
        self.spread_rate = 1  # The increment per step to increase self.spread by
        self.spread_threshold = 1
        self.spread = 0  # When equal or greater than spread_threshold, the smoke will spread to its neighbors

    def step(self):
        if self.spread >= 1:
            smoke_neighborhood = self.model.grid.get_neighborhood(
                self.pos, moore=False, include_center=False, radius=self.smoke_radius
            )
            for neighbor in smoke_neighborhood:
                place_smoke = True
                contents = self.model.grid.get_cell_list_contents(neighbor)
                for agent in contents:
                    if not agent.spreads_smoke:
                        place_smoke = False
                        break

                if place_smoke:
                    smoke = Smoke(neighbor, self.model)
                    self.model.grid.place_agent(smoke, neighbor)
                    self.model.schedule.add(smoke)

        if self.spread >= self.spread_threshold:
            self.spread_rate = 0
        else:
            self.spread += self.spread_rate

    def get_position(self):
        return self.pos


class DeadHuman(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=True, flammable=True, spreads_smoke=True, model=model)


class Human(Agent):
    """
    A human agent, which will attempt to escape from the grid.

    Attributes:
        ID: Unique identifier of the Agent
        Position (x,y): Position of the agent on the Grid
        Health: Health of the agent (between 0 and 1)
        ...
    """

    # 行动能力
    class Mobility(IntEnum):
        INCAPACITATED = 0
        NORMAL = 1
        PANIC = 2

    # 生存状态
    class Status(IntEnum):
        DEAD = 0
        ALIVE = 1
        ESCAPED = 2

    # 计划执行的行为
    class Action(IntEnum):
        PHYSICAL_SUPPORT = 0
        MORALE_SUPPORT = 1
        VERBAL_SUPPORT = 2
        RETREAT = 3

    MIN_HEALTH = 0.0  # 最小生命值
    MAX_HEALTH = 1.0  # 最大生命值

    MIN_EXPERIENCE = 1  # 最小经验值
    MAX_EXPERIENCE = 10  # 最大经验值

    MIN_SPEED = 0.0  # 最小速度
    MAX_SPEED = 2.0  # 最大速度

    MIN_KNOWLEDGE = 0  # 最小知识量
    MAX_KNOWLEDGE = 1  # 最大知识量

    MAX_SHOCK = 1.0  # 最小惊吓度
    MIN_SHOCK = 0.0  # 最大惊吓度
    # Shock modifiers when encountering certain objects per object, per step
    DEFAULT_SHOCK_MODIFIER = -0.1  # 默认每步惊吓值降低量
    SHOCK_MODIFIER_DEAD_HUMAN = 1.0  # 遇到尸体增加惊吓
    SHOCK_MODIFIER_FIRE = 0.2  # 遇到火焰增加惊吓
    SHOCK_MODIFIER_SMOKE = 0.05  # 遇到烟雾增加惊吓
    SHOCK_MODIFIER_AFFECTED_HUMAN = 0.1  # 遇到受伤人增加惊吓

    # The value the panic score must reach for an agent to start panic behaviour
    PANIC_THRESHOLD = 0.8  # 惊吓值达到此阈值将引发恐慌

    HEALTH_MODIFIER_FIRE = 0.2  # 接触火焰对生命值的影响
    HEALTH_MODIFIER_SMOKE = 0.005  # 接触烟雾对生命值的影响

    SPEED_MODIFIER_FIRE = 2  # 接触火焰对速度的影响
    SPEED_MODIFIER_SMOKE = 0.1  # 接触烟雾对速度的影响

    # When the health value drops below this value, the agent will being to slow down
    SLOWDOWN_THRESHOLD = 0.5  # 生命值低于该值将减速

    MIN_PUSH_DAMAGE = 0.01  # 推搡造成的最小伤害
    MAX_PUSH_DAMAGE = 0.2  # 推搡造成的最大伤害

    def __init__(
            self,
            pos: Coordinate,  # 初始位置
            health: float,  # 初始生命值
            speed: float,  # 初始速度
            vision: int,  # 视觉半径
            collaborates: bool,  # 是否合作
            nervousness,  # 神经质程度（越大越容易惊慌）
            experience,  # 经验值
            believes_alarm: bool,  # 是否相信警报
            model,  # 仿真模型引用
    ):
        rand_id = get_random_id()
        super().__init__(rand_id, model)

        self.traversable = False  # 是否可以被穿越

        self.flammable = True  # 是否易燃
        self.spreads_smoke = True  # 是否散播烟雾

        self.pos = pos
        self.visibility = 2
        self.health = health
        self.mobility: Human.Mobility = Human.Mobility.NORMAL
        self.shock: float = self.MIN_SHOCK
        self.speed = speed
        self.vision = vision

        self.collaborates = collaborates  # 是否尝试合作

        self.verbal_collaboration_count: int = 0
        self.morale_collaboration_count: int = 0
        self.physical_collaboration_count: int = 0

        self.morale_boost: bool = False
        self.carried: bool = False
        self.carrying: Union[Human, None] = None

        self.knowledge = self.MIN_KNOWLEDGE
        self.nervousness = nervousness
        self.experience = experience
        self.believes_alarm = believes_alarm
        self.escaped: bool = False

        # The agent and seen location (agent, (x, y)) the agent is planning to move to
        self.planned_target: tuple[Agent, Coordinate] = (
            None,
            None,
        )

        self.planned_action: Human.Action = None  # 目标agent及坐标

        self.visible_tiles: tuple[Coordinate, tuple[Agent]] = []  # 当前可视范围内容

        self.known_tiles: dict[Coordinate, set[Agent]] = {}  # 已知地图信息

        self.visited_tiles: set[Coordinate] = {self.pos}  # 已访问位置

    def update_sight_tiles(self, visible_neighborhood):
        """更新智能体视野中可见的位置，并在网格上标记这些位置"""
        if len(self.visible_tiles) > 0:
            for pos, _ in self.visible_tiles:
                contents = self.model.grid.get_cell_list_contents(pos)
                for agent in contents:
                    if isinstance(agent, Sight):
                        self.model.grid.remove_agent(agent)

        # Add new vision tiles
        for contents, tile in visible_neighborhood:
            # Don't place if the tile has contents but the agent can't see it
            if self.model.grid.is_cell_empty(tile) or len(contents) > 0:
                sight_object = Sight(tile, self.model)
                self.model.grid.place_agent(sight_object, tile)

    def get_visible_tiles(self) -> tuple[Coordinate, tuple[Agent]]:
        """
        根据当前坐标，通过一种简化射线投射算法(Bresenham Line)模拟视觉感知，包括障碍物、能见度、可视性分数（visibility）等影响
        """
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=self.vision
        )
        visible_neighborhood = set()

        # A set of already checked tiles, for avoiding repetition and thus increased efficiency
        checked_tiles = set()

        # 获取视觉半径范围内的邻近格子
        for pos in reversed(neighborhood):
            if pos not in checked_tiles:
                blocked = False
                try:
                    smoke_count = 0  # 这条路径上smoke的数量
                    path = get_line(self.pos, pos)

                    for i, tile in enumerate(path):
                        contents = self.model.grid.get_cell_list_contents(tile)
                        visible_contents = []
                        for obj in contents:
                            if isinstance(obj, Sight):
                                # 忽略可视物体
                                continue
                            elif isinstance(obj, Wall):
                                # 遇到墙壁，设置blocked为True，不在继续进行
                                blocked = True
                                break
                            elif isinstance(obj, Smoke):
                                # 遇到烟雾，增加smoke数量
                                smoke_count += 1

                            # 如果visibility > smoke_count，那么就可以看得到
                            if obj.visibility and obj.visibility > smoke_count:
                                visible_contents.append(obj)

                        if blocked:
                            checked_tiles.update(
                                path[i:]
                            )  # 将路径上剩余的物品添加到checked_tiles中，说明看不到
                            break
                        else:
                            # 如果没有阻挡，将当前物品添加到checked_tiles中
                            checked_tiles.add(
                                tile
                            )
                            visible_neighborhood.add((tile, tuple(visible_contents)))

                except Exception as e:
                    print(e)

        if self.model.visualise_vision:
            self.update_sight_tiles(visible_neighborhood)

        return tuple(visible_neighborhood)


    def get_random_target(self, allow_visited=True):
        """随机获取目标点位"""
        graph_nodes = self.model.graph.nodes()

        known_pos = set(self.known_tiles.keys())

        # 从可用点位中排除我们已经访问的点位
        if not allow_visited:
            known_pos -= self.visited_tiles

        # 获取可以被穿越的点位
        traversable_pos = [pos for pos in known_pos if self.location_is_traversable(pos)]

        while not self.planned_target[1]:
            i = np.random.choice(len(traversable_pos))
            target_pos = traversable_pos[i]
            if target_pos in graph_nodes and target_pos != self.pos:
                self.planned_target = (None, target_pos)

    def attempt_exit_plan(self):
        """尝试获取出口位置"""
        self.planned_target = (None, None)
        fire_exits = set()

        for pos, agents in self.known_tiles.items():
            for agent in agents:
                if isinstance(agent, FireExit):
                    fire_exits.add((agent, pos))

        if len(fire_exits) > 0:
            if len(fire_exits) > 1:  # 如果安全出口不止一个
                best_distance = None
                for exit, exit_pos in fire_exits:
                    length = len(
                        get_line(self.pos, exit_pos)
                    )  # 使用Bresenham算法计算一个最近的出口
                    if not best_distance or length < best_distance:
                        best_distance = length
                        self.planned_target = (exit, exit_pos)

            else:
                self.planned_target = fire_exits.pop()
        else:
            # 如果发生火灾，但眼前没有安全出口，首先尝试去到最近的门，如果视野内没有门，则随机移动
            # TODO： 可以添加跟随机制，使得agent可以在这种情况下跟随其他智能体
            found_door = False
            for pos, contents in self.visible_tiles:
                for agent in contents:
                    if isinstance(agent, Door):
                        found_door = True
                        self.planned_target = (agent, pos)
                        break

                if found_door:
                    break

            # 随机选取一个点作为目标
            if not self.planned_target[1]:
                self.get_random_target(allow_visited=False)

    def get_panic_score(self):
        """计算恐慌度，由健康情况、经验值以及惊恐度有关"""
        health_component = 1 / np.exp(self.health / self.nervousness)
        experience_component = 1 / np.exp(self.experience / self.nervousness)

        # 取平均作为新的恐慌度
        panic_score = (health_component + experience_component + self.shock) / 3

        return panic_score

    def incapacitate(self):
        self.stop_carrying()
        self.mobility = Human.Mobility.INCAPACITATED
        self.traversable = True

    def die(self):
        # Store the agent's position of death so we can remove them and place a DeadHuman
        pos = self.pos
        self.model.grid.remove_agent(self)
        dead_self = DeadHuman(pos, self.model)
        self.model.grid.place_agent(dead_self, pos)
        print("Agent died at", pos)

    def health_mobility_rules(self):
        """
        生命值、移动能力判断
        """
        moore_neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=1
        )  # 获取半径为1包括对角线的邻居
        contents = self.model.grid.get_cell_list_contents(moore_neighborhood)

        for agent in contents:
            if isinstance(agent, Fire):
                self.health -= self.HEALTH_MODIFIER_FIRE
                self.speed -= self.SPEED_MODIFIER_FIRE
            elif isinstance(agent, Smoke):
                self.health -= self.HEALTH_MODIFIER_SMOKE

                # 当生命值小于SLOWDOWN_THRESHOLD，开始减速
                if self.health < self.SLOWDOWN_THRESHOLD:
                    self.speed -= self.SPEED_MODIFIER_SMOKE

        # 避免速度和生命值小于0
        if self.health < self.MIN_HEALTH:
            self.health = self.MIN_HEALTH
        if self.speed < self.MIN_SPEED:
            self.speed = self.MIN_SPEED

        # 如果生命值到达最小，则死亡
        if self.health == self.MIN_HEALTH:
            self.stop_carrying()
            self.die()
        # 如果速度达到最小值，则死亡
        elif self.speed == self.MIN_SPEED:
            self.incapacitate()

    def panic_rules(self):
        """修改agent的恐慌度，并根据最新的恐慌度，修改agent的移动速度"""
        if self.morale_boost:  # 如果agent士气受到大涨，他们就不再会恐慌
            return

        # 如果没有新的惊慌惊慌度将会降低
        shock_modifier = self.DEFAULT_SHOCK_MODIFIER
        for _, agents in self.visible_tiles:
            for agent in agents:
                if isinstance(agent, Fire):
                    shock_modifier += self.SHOCK_MODIFIER_FIRE - self.DEFAULT_SHOCK_MODIFIER
                if isinstance(agent, Smoke):
                    shock_modifier += self.SHOCK_MODIFIER_SMOKE - self.DEFAULT_SHOCK_MODIFIER
                if isinstance(agent, DeadHuman):
                    shock_modifier += self.SHOCK_MODIFIER_DEAD_HUMAN - self.DEFAULT_SHOCK_MODIFIER
                if isinstance(agent, Human) and agent.get_mobility() != Human.Mobility.NORMAL:
                    shock_modifier += (
                            self.SHOCK_MODIFIER_AFFECTED_HUMAN - self.DEFAULT_SHOCK_MODIFIER
                    )

        # 如果惊慌度增加，则过去不相信警报的也会相信警报
        if not self.believes_alarm and shock_modifier != self.DEFAULT_SHOCK_MODIFIER:
            print("Agent now believes the fire is real!")
            self.believes_alarm = True

        self.shock += shock_modifier

        # 保证惊慌度在0到1之间
        if self.shock > self.MAX_SHOCK:
            self.shock = self.MAX_SHOCK
        elif self.shock < self.MIN_SHOCK:
            self.shock = self.MIN_SHOCK

        # 更新恐慌度
        panic_score = self.get_panic_score()

        if panic_score >= self.PANIC_THRESHOLD:
            # 大于恐慌度阈值，停止携带他人，增加运动速度
            print("Agent is panicking! Score:", panic_score, "Shock:", self.shock)
            self.stop_carrying()
            self.mobility = Human.Mobility.PANIC

            # 如果一个agent到达恐慌，则会忘记所有前置知识，以及周围的事物，直到没有出现恐慌
            self.known_tiles = {}
            self.knowledge = 0
        elif panic_score < self.PANIC_THRESHOLD and self.mobility == Human.Mobility.PANIC:
            # TODO: 是否可以不修改移动速度（或者与人的体力、健康度有关）
            # 当agent的恐慌度小于阈值，但其上一个状态是恐慌，则取消恐慌，速度变为正常（可以改进）
            print("Agent stopped panicking! Score:", panic_score, "Shock:", self.shock)
            self.mobility = Human.Mobility.NORMAL

    def learn_environment(self):
        """学习环境中的情况"""
        # TODO: 是否有更好的方法模拟智能体在这个过程中的学习过程，知识记忆机制
        if self.knowledge < self.MAX_KNOWLEDGE:
            # 如果环境中有东西可以学习
            new_tiles = 0

            for pos, agents in self.visible_tiles:
                if pos not in self.known_tiles.keys():
                    # 如果视野中的格子不在知识库中，则添加
                    new_tiles += 1
                self.known_tiles[pos] = set(agents)

            # 相应地更新知识属性
            total_tiles = self.model.grid.width * self.model.grid.height
            new_knowledge_percentage = new_tiles / total_tiles
            self.knowledge = self.knowledge + new_knowledge_percentage
            # print("Current knowledge:", self.knowledge)

    def get_collaboration_cost(self):
        panic_score = self.get_panic_score()
        total_count = (
                self.verbal_collaboration_count
                + self.morale_collaboration_count
                + self.physical_collaboration_count
        )

        collaboration_component = 1 / np.exp(
            1 / (total_count + 1)
        )  # The more time this agent has collaborated, the higher the score will become
        collaboration_cost = (collaboration_component + panic_score) / 2
        # print("Collaboration cost:", collaboration_cost, "Component:", collaboration_component, "Panic component:", panic_score)

        return collaboration_cost

    def test_collaboration(self) -> bool:
        collaboration_cost = self.get_collaboration_cost()

        rand = np.random.random()
        # TODO: 可以添加agent之间的关系（父母妻女），强调某些合作的必要性
        # 如果随机数大于我们的合作成本，则进行合作（合作成本越高，合作的可能性越小）
        if rand > collaboration_cost:
            return True
        else:
            return False

    def verbal_collaboration(self, target_agent: Self, target_location: Coordinate):
        """口头合作：告知别人出口在什么位置"""
        success = False
        for _, agents in self.visible_tiles:
            for agent in agents:
                if isinstance(agent, Human) and agent.get_mobility() == Human.Mobility.NORMAL:
                    if not agent.believes_alarm:
                        agent.set_believes(True)

                    # 告知agent目标位置
                    if not target_location in agent.known_tiles:
                        agent.known_tiles[target_location] = set()

                    agent.known_tiles[target_location].add(target_agent)
                    success = True

        if success:
            print("Agent informed others of a fire exit!")
            self.verbal_collaboration_count += 1

    def check_for_collaboration(self):
        """检查是否有合作的可能，包括携带他人以及口头合作"""
        # 如果一个智能体已经携带了一个人，则不再进行搜寻
        if self.carrying:
            return

        if self.test_collaboration():
            for location, visible_agents in self.visible_tiles:
                if self.planned_action:
                    break

                for agent in visible_agents:
                    if isinstance(agent, Human) and not self.planned_action:
                        if agent.get_mobility() == Human.Mobility.INCAPACITATED:
                            # 如果这个agent失去行为能力，则首先到达他的位置，然后携带他
                            self.planned_target = (
                                agent,
                                location,
                            )
                            # 携带agent
                            self.planned_action = Human.Action.PHYSICAL_SUPPORT
                            # print("Agent planned physical collaboration at", location)
                            break
                        elif (
                                agent.get_mobility() == Human.Mobility.PANIC
                                and not self.planned_action
                        ):
                            # 尝试跟随agent
                            self.planned_target = (
                                agent,
                                location,
                            )
                            # Plan to do morale collaboration with the agent
                            self.planned_action = Human.Action.MORALE_SUPPORT
                            # print("Agent planned morale collaboration at", location)
                            break
                    elif isinstance(agent, FireExit):
                        # Verbal collaboration
                        self.verbal_collaboration(agent, location)

    def get_next_location(self, path):
        path_length = len(path)
        speed_int = int(np.round(self.speed))

        try:
            if path_length <= speed_int:
                next_location = path[path_length - 1]
            else:
                next_location = path[speed_int]

            next_path = []
            for location in path:
                next_path.append(location)
                if location == next_location:
                    break

            return (next_location, next_path)
        except Exception as e:
            raise Exception(
                f"Failed to get next location: {e}\nPath: {path},\nlen: {length},\nSpeed: {self.speed}"
            )

    def get_path(self, graph, target, include_target=True) -> list[Coordinate]:
        path = []
        visible_tiles_pos = [pos for pos, _ in self.visible_tiles]

        try:
            if target in visible_tiles_pos:  # Target is visible, so simply take the shortest path
                path = nx.shortest_path(graph, self.pos, target)
            else:  # Target is not visible, so do less efficient pathing
                # TODO: In the future this could be replaced with a more naive path algorithm
                path = nx.shortest_path(graph, self.pos, target)

                if not include_target:
                    del path[
                        -1
                    ]  # We don't want the target included in the path, so delete the last element

            return list(path)
        except nx.exception.NodeNotFound as e:
            graph_nodes = graph.nodes()

            if target not in graph_nodes:
                contents = self.model.grid.get_cell_list_contents(target)
                print(f"Target node not found! Expected {target}, with contents {contents}")
                return path
            elif self.pos not in graph_nodes:
                contents = self.model.grid.get_cell_list_contents(self.pos)
                raise Exception(
                    f"Current position not found!\nPosition: {self.pos},\nContents: {contents}"
                )
            else:
                raise e

        except nx.exception.NetworkXNoPath as e:
            print(f"No path between nodes! ({self.pos} -> {target})")
            return path

    def location_is_traversable(self, pos) -> bool:
        """判断某一点位上的agent是否可以被穿过"""
        if not self.model.grid.is_cell_empty(pos):
            contents = self.model.grid.get_cell_list_contents(pos)
            for agent in contents:
                if not agent.traversable:
                    return False

        return True

    def get_retreat_location(self, next_location) -> Coordinate:
        x, y = self.pos
        next_x, next_y = next_location
        diff_x = x - next_x
        diff_y = y - next_y

        retreat_location = (sum([x, diff_x]), sum([y, diff_y]))
        return retreat_location

    def check_retreat(self, next_path, next_location) -> bool:
        # Get the contents of any visible locations in the next path
        visible_path = []
        for visible_pos, _ in self.visible_tiles:
            if visible_pos in next_path:
                visible_path.append(visible_pos)

        visible_contents = self.model.grid.get_cell_list_contents(visible_path)
        for agent in visible_contents:
            if (isinstance(agent, Smoke) and not self.planned_action) or isinstance(agent, Fire):
                # There's a danger in the visible path, so try and retreat in the opposite direction
                # Retreat if there's fire, or smoke (and no collaboration attempt)
                retreat_location = self.get_retreat_location(next_location)

                # Check if retreat location is out of bounds
                if not self.model.grid.out_of_bounds(retreat_location):
                    # Check if the retreat location is also smoke, if so, we are surrounded by smoke, so move randomly
                    contents = self.model.grid.get_cell_list_contents(retreat_location)
                    for agent in contents:
                        if isinstance(agent, Smoke) or isinstance(agent, Fire):
                            self.get_random_target()
                            print("Agent surrounded by smoke and moving randomly")
                            retreat_location = None
                            break

                    if retreat_location:
                        print("Agent retreating opposite to fire/smoke")
                        self.planned_target = (None, retreat_location)
                else:
                    self.get_random_target()  # Since our retreat is out of bounds, just go to a random location

                self.planned_action = Human.Action.RETREAT
                return True

        return False

    def update_target(self):
        """更新目标agent的位置，防止agent发生了移动或已经死亡"""
        planned_agent = self.planned_target[0]
        if planned_agent:
            current_pos = planned_agent.get_position()
            if current_pos and current_pos != self.planned_target[1]:  # 目标agent移动
                self.planned_target = (planned_agent, current_pos)
            elif not current_pos:  # 目标agent不存在了
                self.planned_target = (None, None)
                self.planned_action = None

    def update_action(self):
        """更新行为"""
        planned_agent, _ = self.planned_target

        if planned_agent:
            # 智能体曾计划进行合作，但其不再恐慌或不在存活后，不在进行合作
            if self.planned_action == Human.Action.MORALE_SUPPORT and (
                    planned_agent.get_mobility() != Human.Mobility.PANIC
                    or not planned_agent.get_status() == Human.Status.ALIVE
            ):
                # print("Target agent no longer panicking. Dropping action.")
                self.planned_target = (None, None)
                self.planned_action = None
            # 智能体曾计划携带某智能体，但待携带智能体不在丧失行动能力或已被抬走或死亡，则不再携带
        elif self.planned_action == Human.Action.PHYSICAL_SUPPORT and (
                (planned_agent.get_mobility() != Human.Mobility.INCAPACITATED)
                or planned_agent.is_carried()
                or planned_agent.get_status() != Human.Status.ALIVE
        ):
            self.planned_target = (None, None)
            self.planned_action = None
        elif self.planned_action == Human.Action.RETREAT:
            return
        else:  # Can no longer perform the action
            self.planned_target = (None, None)
            self.planned_action = None

    def perform_action(self):
        agent, _ = self.planned_target

        if self.planned_action == Human.Action.PHYSICAL_SUPPORT:
            if not agent.is_carried():
                self.carrying = agent
                agent.set_carried(True)
                self.physical_collaboration_count += 1
                print("Agent started carrying another agent")
        elif self.planned_action == Human.Action.MORALE_SUPPORT:
            # Attempt to give the agent a permanent morale boost according to your experience score
            if agent.attempt_morale_boost(self.experience):
                print("Morale boost succeeded")
            else:
                print("Morale boost failed")

            self.morale_collaboration_count += 1

        self.planned_action = None

    def push_human_agent(self, agent: Self):
        # push the agent to a random 1 square away traversable Coordinate
        neighborhood = self.model.grid.get_neighborhood(
            agent.get_position(),
            moore=True,
            include_center=False,
            radius=1,
        )
        traversable_neighborhood = [
            neighbor_pos
            for neighbor_pos in neighborhood
            if self.location_is_traversable(neighbor_pos)
        ]

        if len(traversable_neighborhood) > 0:
            # push the human agent to a random traversable position
            i = np.random.choice(len(traversable_neighborhood))
            push_pos = traversable_neighborhood[i]
            print(
                f"Agent {self.unique_id} pushed agent {agent.unique_id} from {agent.pos} to {push_pos}"
            )
            self.model.grid.move_agent(agent, push_pos)

            # inure the pushed agent slightly
            current_health = agent.get_health()
            damage = np.random.uniform(self.MIN_PUSH_DAMAGE, self.MAX_PUSH_DAMAGE)
            agent.set_health(current_health - damage)
        else:
            neighborhood_contents = {}
            for pos in neighborhood:
                neighborhood_contents[pos] = self.model.grid.get_cell_list_contents(pos)
            print(
                f"Could not push agent due to no traversable locations.\nNeighborhood Contents: {neighborhood_contents}"
            )

    def move_toward_target(self):
        next_location: Coordinate = None
        pruned_edges = set()
        graph = deepcopy(self.model.graph)

        self.update_target()  # 更新目标点位
        if self.planned_action:  # 更新后续行为
            self.update_action()

        while self.planned_target[1] and not next_location:
            if self.location_is_traversable(self.planned_target[1]):
                # Target is traversable
                path = self.get_path(graph, self.planned_target[1])
            else:
                # Target is not traversable (e.g. we are going to another Human), so don't include target in the path
                path = self.get_path(graph, self.planned_target[1], include_target=False)

            if len(path) > 0:
                next_location, next_path = self.get_next_location(path)

                if next_location == self.pos:
                    continue

                if next_location == None:
                    raise Exception("Next location can't be none")

                if self.check_retreat(next_path, next_location):
                    # We are retreating and therefore need to try a totally new path, so continue from the start of the loop
                    continue

                # Test the next location to see if we can move there
                if self.location_is_traversable(next_location):
                    # Move normally
                    self.previous_pos = self.pos
                    self.model.grid.move_agent(self, next_location)
                    self.visited_tiles.add(next_location)

                    if self.carrying:
                        agent = self.carrying
                        if agent.get_status() == Human.Status.DEAD:
                            # Agent is dead, so we can't carry them any more
                            self.stop_carrying()
                        else:
                            # Agent is alive, so try to move them
                            try:
                                self.model.grid.move_agent(self.carrying, self.pos)
                            except Exception as e:
                                agent = self.carrying
                                raise Exception(
                                    f"Failed to move carried agent:\nException:{e}\nAgent: {agent}\nAgent Position: {agent.get_position()}\nSelf Agent Positon: {self.pos}"
                                )

                elif self.pos == path[-1]:
                    # The human reached their target!

                    if self.planned_action:
                        self.perform_action()

                    self.planned_target = (None, None)
                    self.planned_action = None
                    break

                else:
                    # We want to move here but it's blocked

                    # check if the location is blocked due to a Human agent
                    pushed = False
                    contents = self.model.grid.get_cell_list_contents(next_location)
                    for agent in contents:
                        # Test the panic value to see if this agent "pushes" the blocking agent aside
                        if (
                                isinstance(agent, Human)
                                and agent.mobility != Human.Mobility.INCAPACITATED
                        ) and (
                                (
                                        self.get_panic_score() >= self.PANIC_THRESHOLD
                                        and self.mobility == Human.Mobility.NORMAL
                                )
                                or self.mobility == Human.Mobility.PANIC
                        ):
                            # push the agent and then move to the next_location
                            self.push_human_agent(agent)
                            self.previous_pos = self.pos
                            self.model.grid.move_agent(self, next_location)
                            self.visited_tiles.add(next_location)
                            pushed = True
                            break
                    if pushed:
                        continue

                    # Remove the next location from the temporary graph so we can try pathing again without it
                    edges = graph.edges(next_location)
                    pruned_edges.update(edges)
                    graph.remove_node(next_location)

                    # Reset planned_target if the next location was the end of the path
                    if next_location == path[-1]:
                        next_location = None
                        self.planned_target = (None, None)
                        self.planned_action = None
                        break
                    else:
                        next_location = None

            else:  # No path is possible, so drop the target
                self.planned_target = (None, None)
                self.planned_action = None
                break

        if len(pruned_edges) > 0:
            # Add back the edges we removed when removing any non-traversable nodes from the global graph, because they may be traversable again next step
            graph.add_edges_from(list(pruned_edges))

    def step(self):
        if not self.escaped and self.pos:
            self.health_mobility_rules()

            if self.mobility == Human.Mobility.INCAPACITATED or not self.pos:
                # 已死亡，则该agent停止进程
                return

            # 更新视野内的物体
            self.visible_tiles = self.get_visible_tiles()

            # 修改agent的恐慌度，并根据最新的恐慌度，修改agent的移动速度
            self.panic_rules()

            # 学习环境中的情况
            self.learn_environment()

            # 目标agent（可能是安全出口，也可以是人类智能体）
            planned_target_agent = self.planned_target[0]

            # 如果火灾已经发生，并且agent认为已经发生火灾，则尝试规划出口位置（如果尚未规划出口位置且为采取行动）
            if self.model.fire_started and self.believes_alarm:
                if not isinstance(planned_target_agent, FireExit) and not self.planned_action:
                    self.attempt_exit_plan()

                # 如果智能体的移动速度为正常，检查视野内是否可以进行合作
                if self.mobility == Human.Mobility.NORMAL and self.collaborates:
                    self.check_for_collaboration()

            # 目标点位
            planned_pos = self.planned_target[1]
            if not planned_pos:
                self.get_random_target()
            elif self.mobility == Human.Mobility.PANIC:
                # 判断恐慌度以及是否会晕倒
                panic_score = self.get_panic_score()

                if panic_score > 0.9 and np.random.random() < panic_score:
                    # 如果他的恐慌度达到90%，则可能会晕倒
                    print("Agent fainted!")
                    self.incapacitate()
                    return

            self.move_toward_target()

            # 智能体到达安全出口，结束进程
            if self.model.fire_started and self.pos in self.model.fire_exits.keys():
                # 携带的智能体也结束进程
                if self.carrying:
                    carried_agent = self.carrying
                    carried_agent.escaped = True
                    self.model.grid.remove_agent(carried_agent)

                self.escaped = True
                self.model.grid.remove_agent(self)

    def get_status(self):
        if self.health > self.MIN_HEALTH and not self.escaped:
            return Human.Status.ALIVE
        elif self.health <= self.MIN_HEALTH and not self.escaped:
            return Human.Status.DEAD
        elif self.escaped:
            return Human.Status.ESCAPED

        return None

    def get_speed(self):
        return self.speed

    def get_mobility(self):
        return self.mobility

    def get_health(self):
        return self.health

    def get_position(self):
        return self.pos

    def get_plan(self):
        return (self.planned_target, self.planned_action)

    def set_plan(self, agent, location):
        self.planned_action = None
        self.planned_target = (agent, location)

    def set_health(self, value: float):
        self.health = value

    def set_believes(self, value: bool):
        """设置agent相信警报"""
        if value and not self.believes_alarm:
            print("Agent told to believe the alarm!")

        self.believes_alarm = value

    def attempt_morale_boost(self, experience: int):
        rand = np.random.random()
        if rand < (experience / self.MAX_EXPERIENCE):
            self.morale_boost = True
            self.mobility = Human.Mobility.NORMAL
            return True
        else:
            return False

    def stop_carrying(self):
        """停止携带他人"""
        if self.carrying:
            carried_agent = self.carrying
            carried_agent.set_carried(False)
            self.carrying = None
            self.planned_action = None
            print("Agent stopped carrying another agent")

    def set_carried(self, value: bool):
        self.carried = value

    def is_carried(self):
        return self.carried

    def is_carrying(self):
        if self.carrying:
            return True
        else:
            return False

    def get_verbal_collaboration_count(self):
        return self.verbal_collaboration_count

    def get_morale_collaboration_count(self):
        return self.morale_collaboration_count

    def get_physical_collaboration_count(self):
        return self.physical_collaboration_count
