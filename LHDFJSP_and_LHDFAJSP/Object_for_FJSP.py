class Object:
    def __init__(self, I):
        self.I = I  # 工件/机器编号
        self.Start = []  # 工件/机器加工开始时间
        self.End = []  # 工件/机器加工结束时间
        self.Transport = []  # 工件运输时间
        self.T = []  # 工件/机器加工时间
        self.assign_for = []  # 分配

    def _add(self, S, E, obs, pt, trans,):  # 运输时间
        # obs:安排的对象
        self.Start.append(S)
        self.End.append(E)
        self.Start.sort()
        self.End.sort()
        self.Transport.append(trans)
        self.T.append(pt)
        self.assign_for.insert(self.End.index(E), obs)

    def idle_time(self):  # 机器空闲时间 format: [[0,3],[10,15]....]
        Idle = []
        try:
            if self.Start[0] != 0:
                Idle.append([0, self.Start[0]])
            K = [[self.End[i], self.Start[i + 1]] for i in range(len(self.End) - 1) if self.Start[i + 1] - self.End[i] > 0]
            Idle.extend(K)
        except:
            pass
        return Idle

