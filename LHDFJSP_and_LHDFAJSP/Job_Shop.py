import numpy as np
import random
from Instance_Generator import Processing_time, A, D, M_num, Op_num, J, O_num, J_num, jobShop_Dict, transportJS, \
    transportMAC
from Object_for_FJSP import Object
from ActionFunc import ActionMethod


class Situation:
    def __init__(self, J_num, M_num, O_num, J, Processing_time, D, Ai, Op_num, jobShop_Dict, transportJS, transportMAC):
        self.transport_energy = 2
        self.Ai = Ai  # 工件到达时间
        self.D = D  # 交货期
        self.O_num = O_num  # 工序总数
        self.M_num = M_num  # 机器数
        self.J_num = J_num  # 工件数
        self.J = J  # 工件对应的工序数
        self.Op_num = Op_num  # 工序数量
        self.jobShop_Dict = jobShop_Dict  # 机器编号与所属车间,机器加工功率,空载功率的字典
        self.transportJS = transportJS  # 车间运输时间
        self.transportMAC = transportMAC  # 设备运输时间
        self.Processing_time = Processing_time  # 加工时间
        self.JOB_MAC = []
        self.last_MAC = [-1 for i in range(J_num)]
        self.CTK = [0 for i in range(M_num)]  # 各机器上最后一道工序的完工时间列表
        self.NPO = [0 for i in range(J_num)]  # 各工件的已加工工序数列表
        self.UR = [0 for i in range(M_num)]  # 各机器的实际使用率
        self.CRJ = [0 for i in range(J_num)]  # 工件完工率
        # -----------------------------------------------------------
        self.Jobs = []  # 工件集：存储工件对象（Object_for_FJSP）
        for i in range(J_num):
            F = Object(i)
            self.Jobs.append(F)
        # -----------------------------------------------------------
        self.Machines = []  # 机器集：存储工件对象（Object_for_FJSP）
        for i in range(M_num):
            F = Object(i)
            self.Machines.append(F)

    # 更新数据
    def _Update(self, Job, Machine, seq_job):
        self.CTK[Machine] = max(self.Machines[Machine].End)
        self.NPO[Job] += 1
        try:
            self.UR[Machine] = sum(self.Machines[Machine].T) / self.Machines[Machine].End[-1]
        except:
            self.UR[Machine] = 0
        # print('update', self.UR)
        self.CRJ[Job] = self.NPO[Job] / self.J[Job][0]
        self.JOB_MAC.append(seq_job)  # 加工顺序
        # Seq_Job = Job, O_n, Machine, PT, transTime

    # 机器平均使用率
    def Features(self):  # 状态特征
        # 1 机器平均利用率
        UR_ave = sum(self.UR) / self.M_num
        K = 0
        for ur in self.UR:
            K += np.square(ur - UR_ave)
        # 2 机器利用率标准差
        U_std = np.sqrt(K / self.M_num)
        # 3 工序平均完成率
        CRO_ave = sum(self.NPO) / self.O_num
        # 4 工件平均完成率
        CRJ_ave = sum(self.CRJ) / self.J_num
        K = 0
        for ur in self.CRJ:
            K += np.square(ur - CRJ_ave)
        # 5 工件完成率标准差
        CRJ_std = np.sqrt(K / self.J_num)

        # 6 预计拖期率 Tard_e
        T_cur = sum(self.CTK) / self.M_num
        N_tard, N_left = 0, 0  # N_tard：延迟工序； N_left：未完成工序
        for i in range(self.J_num):
            if J[i][0] > self.NPO[i]:
                N_left += self.J[i][0] - self.NPO[i]
                T_left = 0
                for j in range(self.NPO[i], J[i][0]):
                    M_ij = [k for k in self.Processing_time[i][j] if k > 0 or k < 999]
                    T_left += sum(M_ij) / len(M_ij)  # 工序平均加工时间
                    try:
                        EDT = max(T_cur, self.Jobs[i].End[-1]) + T_left - self.D[i]
                    except:
                        EDT = T_cur + T_left - self.D[i]
                    if EDT > 0:  # <可以考虑添加该机器的平均运输时间>
                        N_tard += self.J[i][0] - j + 1
        try:
            Tard_e = N_tard / N_left
        except:
            Tard_e = 9999

        # 7 实际拖期率 Tard_a
        N_tard, N_left = 0, 0  # N_tard：延迟工序； N_left：未完成工序
        for i in range(self.J_num):
            if J[i][0] > self.NPO[i]:
                N_left += self.J[i][0] - self.NPO[i]
                for j in range(self.NPO[i], J[i][0]):
                    try:
                        if self.Jobs[i].End[-1] > self.D[i]:
                            N_tard += self.J[i][0] - j
                    except:
                        pass
        try:
            Tard_a = N_tard / N_left
        except:
            Tard_a = 9999

        # 8 预计加权拖期率 wTrad_e
        N_tard, N_left = 0, 0  # N_tard：延迟工序； N_left：未完成工序
        Ti_ave_tard = 0
        Ti_ave_uncom = 0

        for i in range(self.J_num):
            if J[i][0] > self.NPO[i]:
                N_left += self.J[i][0] - self.NPO[i]
                Process_left = 0
                Trans_left = 0
                for j in range(self.NPO[i], J[i][0]):
                    M_ij = [k for k in self.Processing_time[i][j] if k > 0 or k < 999]
                    Process_left += sum(M_ij) / len(M_ij)  # 工序平均加工时间

                    # 可加工O_ij机器
                    M_use = [k for k in range(len(self.Processing_time[i][j])) if self.Processing_time[i][j][k] != -1]
                    # 可加工O_ij车间
                    JS_use = []
                    for k in M_use:
                        if self.jobShop_Dict[k][0] not in JS_use:
                            JS_use.append(self.jobShop_Dict[k][0])
                    if self.NPO[i] == 0:
                        Trans_left = 0
                    else:
                        for m in range(len(self.JOB_MAC)):
                            if self.JOB_MAC[m][0] == i and self.JOB_MAC[m][1] == (self.NPO[i] - 1):
                                Trans_left = sum(self.transportJS[self.jobShop_Dict[self.JOB_MAC[m][2]][0]]) / len(
                                    JS_use) * (self.J[i][0] - self.NPO[i] - random.randint(1, 2))
                # 未完成平均时间
                Ti_ave_uncom += (Process_left + Trans_left) * J[i][1]
                try:
                    if not self.Jobs[i].End:
                        EDT = T_cur + Process_left + Trans_left - self.D[i]
                    else:
                        EDT = max(T_cur, self.Jobs[i].End[-1]) + Process_left + Trans_left - self.D[i]
                except:
                    EDT = T_cur + Process_left + + Trans_left - self.D[i]
                if EDT > 0:
                    Ti_ave_tard += (Process_left + Trans_left) * J[i][1]
        try:
            WTard_e = Ti_ave_tard / Ti_ave_uncom
        except:
            WTard_e = 9999

        # 9 能耗指标 EI(t)
        #
        # levelTE = []
        # complete_op = len(self.JOB_MAC)
        if len(self.JOB_MAC) > 1:
            job_front = self.JOB_MAC[-2][0]
        else:
            job_front = -1
        job = self.JOB_MAC[-1][0]
        O_n = self.JOB_MAC[-1][1]
        TE_process = []
        TE_trans = []
        TE_idle = []
        # 预估所有机器加工能耗
        for k in range(len(self.CTK)):
            if self.Processing_time[job][O_n][k] != -1:
                # 加工能耗
                TE_pro = self.Processing_time[job][O_n][k] * self.jobShop_Dict[k][1]
                TE_process.append(TE_pro)
                # 运输能耗
                # if not self.JOB_MAC:
                #     TE_trans.append(0)
                if len(self.JOB_MAC) == 1:
                    TE_trans.append(0)
                elif self.JOB_MAC[-2][2] == k:
                    TE_trans.append(0)
                else:
                    if self.transportMAC[self.JOB_MAC[-2][2]][k] != -1:
                        TE_tra = self.transportMAC[self.JOB_MAC[-2][2]][k] * self.transport_energy
                        TE_trans.append(TE_tra)
                    else:
                        TE_tra = self.transportJS[self.jobShop_Dict[
                            self.JOB_MAC[-2][2]][0]][self.jobShop_Dict[k][0]] * self.transport_energy
                        TE_trans.append(TE_tra)
                # 空载能耗
                try:
                    if job_front == -1:
                        last_ot = 0
                    else:
                        last_ot = max(self.Jobs[job_front].End)  # 上道工序加工完成时间
                except:
                    last_ot = 0
                try:
                    last_mt = max(self.Machines[k].End)  # 机器最后完工时间
                except:
                    last_mt = 0
                # 运输时间
                if not self.JOB_MAC:
                    transTime = 0
                elif len(self.JOB_MAC) == 1:
                    transTime = 0
                elif self.JOB_MAC[-2][2] == k:
                    transTime = 0
                else:
                    if self.transportMAC[self.JOB_MAC[-2][2]][k] != -1:
                        transTime = self.transportMAC[self.JOB_MAC[-2][2]][k]
                    else:
                        transTime = self.transportJS[self.jobShop_Dict[self.JOB_MAC[-2][2]][0]][
                            self.jobShop_Dict[k][0]]
                # 开始时间计算方法
                if last_ot >= last_mt:
                    Start_time = last_ot + transTime
                else:
                    if last_mt > (last_ot + transTime):
                        Start_time = last_mt
                    else:
                        Start_time = last_ot + transTime
                if not self.Machines[k].End:
                    TE_idle.append(0)
                else:
                    if Start_time > self.Machines[k].End[-1]:
                        TE_idl = (Start_time - self.Machines[k].End[-1]) * self.jobShop_Dict[k][2]
                        TE_idle.append(TE_idl)
                    else:
                        TE_idle.append(0)
            else:
                TE_process.append(0)
                TE_trans.append(0)
                TE_idle.append(0)
        TE_ij = [TE_p + TE_t + TE_i for TE_p, TE_t, TE_i in zip(TE_process, TE_trans, TE_idle)]
        # 10 能耗指标
        TE_cur = TE_ij[self.JOB_MAC[-1][2]]
        TE_mid = (max(TE_ij) + min(TE_ij)) / 2
        EI = abs((TE_mid - TE_cur) / (TE_mid - min(TE_ij)))

        # 11-20 设备完成时间 - 当前完成时间均值
        endTime = []
        for M in range(M_num):
            if not self.Machines[M].End:
                endTime.append(0)
            else:
                endTime.append(self.Machines[M].End[-1])
        if not endTime:
            averageEndTime = 0
        else:
            averageEndTime = sum(endTime) / len(endTime)

        if not self.Machines[0].End:
            Mac_0_End = 0 - averageEndTime
        else:
            Mac_0_End = self.Machines[0].End[-1] - averageEndTime

        if not self.Machines[1].End:
            Mac_1_End = 0 - averageEndTime
        else:
            Mac_1_End = self.Machines[1].End[-1] - averageEndTime

        if not self.Machines[2].End:
            Mac_2_End = 0 - averageEndTime
        else:
            Mac_2_End = self.Machines[2].End[-1] - averageEndTime

        if not self.Machines[3].End:
            Mac_3_End = 0 - averageEndTime
        else:
            Mac_3_End = self.Machines[3].End[-1] - averageEndTime

        if not self.Machines[4].End:
            Mac_4_End = 0 - averageEndTime
        else:
            Mac_4_End = self.Machines[4].End[-1] - averageEndTime

        if not self.Machines[5].End:
            Mac_5_End = 0 - averageEndTime
        else:
            Mac_5_End = self.Machines[5].End[-1] - averageEndTime

        if not self.Machines[6].End:
            Mac_6_End = 0 - averageEndTime
        else:
            Mac_6_End = self.Machines[6].End[-1] - averageEndTime

        if not self.Machines[7].End:
            Mac_7_End = 0 - averageEndTime
        else:
            Mac_7_End = self.Machines[7].End[-1] - averageEndTime

        if not self.Machines[8].End:
            Mac_8_End = 0 - averageEndTime
        else:
            Mac_8_End = self.Machines[8].End[-1] - averageEndTime

        if not self.Machines[9].End:
            Mac_9_End = 0 - averageEndTime
        else:
            Mac_9_End = self.Machines[9].End[-1] - averageEndTime

        # if not self.Machines[10].End:
        #     Mac_10_End = 0 - averageEndTime
        # else:
        #     Mac_10_End = self.Machines[10].End[-1] - averageEndTime
        #
        # if not self.Machines[11].End:
        #     Mac_11_End = 0 - averageEndTime
        # else:
        #     Mac_11_End = self.Machines[11].End[-1] - averageEndTime
        #
        # if not self.Machines[12].End:
        #     Mac_12_End = 0 - averageEndTime
        # else:
        #     Mac_12_End = self.Machines[12].End[-1] - averageEndTime
        #
        # if not self.Machines[13].End:
        #     Mac_13_End = 0 - averageEndTime
        # else:
        #     Mac_13_End = self.Machines[13].End[-1] - averageEndTime
        #
        # if not self.Machines[14].End:
        #     Mac_14_End = 0 - averageEndTime
        # else:
        #     Mac_14_End = self.Machines[14].End[-1] - averageEndTime
        #
        # if not self.Machines[15].End:
        #     Mac_15_End = 0 - averageEndTime
        # else:
        #     Mac_15_End = self.Machines[15].End[-1] - averageEndTime
        #
        # if not self.Machines[16].End:
        #     Mac_16_End = 0 - averageEndTime
        # else:
        #     Mac_16_End = self.Machines[16].End[-1] - averageEndTime
        #
        # if not self.Machines[17].End:
        #     Mac_17_End = 0 - averageEndTime
        # else:
        #     Mac_17_End = self.Machines[17].End[-1] - averageEndTime
        #
        # if not self.Machines[18].End:
        #     Mac_18_End = 0 - averageEndTime
        # else:
        #     Mac_18_End = self.Machines[18].End[-1] - averageEndTime
        #
        # if not self.Machines[19].End:
        #     Mac_19_End = 0 - averageEndTime
        # else:
        #     Mac_19_End = self.Machines[19].End[-1] - averageEndTime

        return UR_ave, U_std, CRO_ave, CRJ_ave, CRJ_std, Tard_e, Tard_a, WTard_e, EI, TE_cur, Mac_0_End, Mac_1_End, \
               Mac_2_End, Mac_3_End, Mac_4_End, Mac_5_End, Mac_6_End, Mac_7_End, Mac_8_End, Mac_9_End
               # Mac_10_End, \
               # Mac_11_End, Mac_12_End, Mac_13_End, Mac_14_End, Mac_15_End, Mac_16_End, Mac_17_End, Mac_18_End, Mac_19_End

    # Composite dispatching rule 1-1
    # J1 M1
    # return Job,Machine
    def rule1_1(self):
        # 工件选择规则 1
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_1(self.CTK, self.M_num, self.J_num, self.Jobs,
                                          self.NPO, self.J, self.D, self.Processing_time)
        # 机器选择规则 1
        Machine = ActionSet.Mac_Selection_1(Job_i, self.Jobs, self.Ai, self.CTK, self.Processing_time)
        # print('This is from rule 1-1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 1-2
    # J1 M2
    # return Job,Machine
    def rule1_2(self):
        # 工件选择规则 1
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_1(self.CTK, self.M_num, self.J_num, self.Jobs,
                                          self.NPO, self.J, self.D, self.Processing_time)
        # 机器选择规则 2
        Machine = ActionSet.Mac_Selection_2(Job_i, self.Jobs, self.CTK, self.Processing_time, self.jobShop_Dict,
                                            self.JOB_MAC, self.transportMAC, self.transport_energy, self.transportJS,
                                            self.Machines)
        # print('This is from rule 1-2:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 1-3
    # J1 M3
    # return Job,Machine
    def rule1_3(self):
        # 工件选择规则 1
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_1(self.CTK, self.M_num, self.J_num, self.Jobs,
                                          self.NPO, self.J, self.D, self.Processing_time)
        # 机器选择规则 3
        Machine = ActionSet.Mac_Selection_3(Job_i, self.Jobs, self.UR, self.Processing_time)
        return Job_i, Machine

    # Composite dispatching rule 1-4
    # J1 M4
    # return Job,Machine
    def rule1_4(self):
        # 工件选择规则 1
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_1(self.CTK, self.M_num, self.J_num, self.Jobs,
                                          self.NPO, self.J, self.D, self.Processing_time)
        # 机器选择规则 4
        Machine = ActionSet.Mac_Selection_4(Job_i, self.Jobs, self.M_num, self.Processing_time)
        # print('This is from rule 1-4:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 1-5
    # J1 M5
    # return Job,Machine
    def rule1_5(self):
        # 工件选择规则 1
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_1(self.CTK, self.M_num, self.J_num, self.Jobs,
                                          self.NPO, self.J, self.D, self.Processing_time)
        # 机器选择规则 5
        Machine = ActionSet.Mac_Selection_5(Job_i, self.Jobs, self.CTK, self.M_num, self.Processing_time)
        # print('This is from rule 1-5:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 2-1
    # J2 M1
    # return Job,Machine
    def rule2_1(self):
        # 工件选择规则 2
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_2(self.CTK, self.M_num, self.J_num, self.Jobs, self.NPO, self.J, self.D,
                                          self.Op_num, self.Processing_time, self.JOB_MAC, self.transportJS,
                                          self.jobShop_Dict)
        # 机器选择规则 1
        Machine = ActionSet.Mac_Selection_1(Job_i, self.Jobs, self.Ai, self.CTK, self.Processing_time)
        # print('This is from rule 2-1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 2-2
    # J2 M2
    # return Job,Machine
    def rule2_2(self):
        # 工件选择规则 2
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_2(self.CTK, self.M_num, self.J_num, self.Jobs, self.NPO, self.J, self.D,
                                          self.Op_num, self.Processing_time, self.JOB_MAC, self.transportJS,
                                          self.jobShop_Dict)
        # 机器选择规则 2
        Machine = ActionSet.Mac_Selection_2(Job_i, self.Jobs, self.CTK, self.Processing_time, self.jobShop_Dict,
                                            self.JOB_MAC, self.transportMAC, self.transport_energy, self.transportJS,
                                            self.Machines)
        # print('This is from rule 2-2:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 2-3
    # J2 M3
    # return Job,Machine
    def rule2_3(self):
        # 工件选择规则 2
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_2(self.CTK, self.M_num, self.J_num, self.Jobs, self.NPO, self.J, self.D,
                                          self.Op_num, self.Processing_time, self.JOB_MAC, self.transportJS,
                                          self.jobShop_Dict)
        # 机器选择规则 3
        Machine = ActionSet.Mac_Selection_3(Job_i, self.Jobs, self.UR, self.Processing_time)
        # print('This is from rule 2-3:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 2-4
    # J2 M4
    # return Job,Machine
    def rule2_4(self):
        # 工件选择规则 2
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_2(self.CTK, self.M_num, self.J_num, self.Jobs, self.NPO, self.J, self.D,
                                          self.Op_num, self.Processing_time, self.JOB_MAC, self.transportJS,
                                          self.jobShop_Dict)
        # 机器选择规则 4
        Machine = ActionSet.Mac_Selection_4(Job_i, self.Jobs, self.M_num, self.Processing_time)
        # print('This is from rule 2-4:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 2-5
    # J2 M5
    # return Job,Machine
    def rule2_5(self):
        # 工件选择规则 2
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_2(self.CTK, self.M_num, self.J_num, self.Jobs, self.NPO, self.J, self.D,
                                          self.Op_num, self.Processing_time, self.JOB_MAC, self.transportJS,
                                          self.jobShop_Dict)
        # 机器选择规则 5
        Machine = ActionSet.Mac_Selection_5(Job_i, self.Jobs, self.CTK, self.M_num, self.Processing_time)
        # print('This is from rule 2-5:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 3-1
    # J3 M1
    # return Job,Machine
    def rule3_1(self):
        # 工件选择规则 3
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_3(self.CTK, self.M_num, self.J_num, self.NPO, self.J,
                                          self.D, self.Processing_time, self.Jobs)
        # 机器选择规则 1
        Machine = ActionSet.Mac_Selection_1(Job_i, self.Jobs, self.Ai, self.CTK, self.Processing_time)
        # print('This is from rule 3-1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 3-2
    # J3 M2
    # return Job,Machine
    def rule3_2(self):
        # 工件选择规则 3
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_3(self.CTK, self.M_num, self.J_num, self.NPO, self.J,
                                          self.D, self.Processing_time, self.Jobs)
        # 机器选择规则 2
        Machine = ActionSet.Mac_Selection_2(Job_i, self.Jobs, self.CTK, self.Processing_time, self.jobShop_Dict,
                                            self.JOB_MAC, self.transportMAC, self.transport_energy, self.transportJS,
                                            self.Machines)
        # print('This is from rule 3-2:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 3-3
    # J3 M3
    # return Job,Machine
    def rule3_3(self):
        # 工件选择规则 3
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_3(self.CTK, self.M_num, self.J_num, self.NPO, self.J,
                                          self.D, self.Processing_time, self.Jobs)
        # 机器选择规则 3
        Machine = ActionSet.Mac_Selection_3(Job_i, self.Jobs, self.UR, self.Processing_time)
        # print('This is from rule 3-3:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 3-4
    # J3 M4
    # return Job,Machine
    def rule3_4(self):
        # 工件选择规则 3
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_3(self.CTK, self.M_num, self.J_num, self.NPO, self.J,
                                          self.D, self.Processing_time, self.Jobs)
        # 机器选择规则 4
        Machine = ActionSet.Mac_Selection_4(Job_i, self.Jobs, self.M_num, self.Processing_time)
        # print('This is from rule 3-4:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 3-5
    # J3 M5
    # return Job,Machine
    def rule3_5(self):
        # 工件选择规则 3
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_3(self.CTK, self.M_num, self.J_num, self.NPO, self.J,
                                          self.D, self.Processing_time, self.Jobs)
        # 机器选择规则 5
        Machine = ActionSet.Mac_Selection_5(Job_i, self.Jobs, self.CTK, self.M_num, self.Processing_time)
        # print('This is from rule 3-5:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 4-1
    # J4 M1
    # return Job,Machine
    def rule4_1(self):
        # 工件选择规则 4
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_4(self.J_num, self.NPO, self.J)
        # 机器选择规则 1
        Machine = ActionSet.Mac_Selection_1(Job_i, self.Jobs, self.Ai, self.CTK, self.Processing_time)
        # print('This is from rule 4-1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 4-2
    # J4 M2
    # return Job,Machine
    def rule4_2(self):
        # 工件选择规则 4
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_4(self.J_num, self.NPO, self.J)
        # 机器选择规则 2
        Machine = ActionSet.Mac_Selection_2(Job_i, self.Jobs, self.CTK, self.Processing_time, self.jobShop_Dict,
                                            self.JOB_MAC, self.transportMAC, self.transport_energy, self.transportJS,
                                            self.Machines)
        # print('This is from rule 4-2:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 4-3
    # J4 M3
    # return Job,Machine
    def rule4_3(self):
        # 工件选择规则 4
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_4(self.J_num, self.NPO, self.J)
        # 机器选择规则 3
        Machine = ActionSet.Mac_Selection_3(Job_i, self.Jobs, self.UR, self.Processing_time)
        # print('This is from rule 4-3:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 4-4
    # J4 M4
    # return Job,Machine
    def rule4_4(self):
        # 工件选择规则 4
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_4(self.J_num, self.NPO, self.J)
        # 机器选择规则 4
        Machine = ActionSet.Mac_Selection_4(Job_i, self.Jobs, self.M_num, self.Processing_time)
        # print('This is from rule 4-4:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 4-5
    # J4 M5
    # return Job,Machine
    def rule4_5(self):
        # 工件选择规则 4
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_4(self.J_num, self.NPO, self.J)
        # 机器选择规则 5
        Machine = ActionSet.Mac_Selection_5(Job_i, self.Jobs, self.CTK, self.M_num, self.Processing_time)
        # print('This is from rule 4-5:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 5-1
    # J5 M1
    # return Job,Machine
    def rule5_1(self):
        # 工件选择规则 5
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_5(self.CTK, self.M_num, self.J_num, self.Jobs, self.NPO, self.J, self.D,
                                          self.CRJ, self.Op_num, self.Processing_time, self.JOB_MAC,
                                          self.transportJS, self.jobShop_Dict)
        # 机器选择规则 1
        Machine = ActionSet.Mac_Selection_1(Job_i, self.Jobs, self.Ai, self.CTK, self.Processing_time)
        # print('This is from rule 5-1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 5-2
    # J5 M2
    # return Job,Machine
    def rule5_2(self):
        # 工件选择规则 5
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_5(self.CTK, self.M_num, self.J_num, self.Jobs, self.NPO, self.J, self.D,
                                          self.CRJ, self.Op_num, self.Processing_time, self.JOB_MAC,
                                          self.transportJS, self.jobShop_Dict)
        # 机器选择规则 2
        Machine = ActionSet.Mac_Selection_2(Job_i, self.Jobs, self.CTK, self.Processing_time, self.jobShop_Dict,
                                            self.JOB_MAC, self.transportMAC, self.transport_energy, self.transportJS,
                                            self.Machines)
        # print('This is from rule 5-2:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 5-3
    # J5 M3
    # return Job,Machine
    def rule5_3(self):
        # 工件选择规则 5
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_5(self.CTK, self.M_num, self.J_num, self.Jobs, self.NPO, self.J, self.D,
                                          self.CRJ, self.Op_num, self.Processing_time, self.JOB_MAC,
                                          self.transportJS, self.jobShop_Dict)
        # 机器选择规则 3
        Machine = ActionSet.Mac_Selection_3(Job_i, self.Jobs, self.UR, self.Processing_time)
        # print('This is from rule 5-3:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 5-4
    # J5 M4
    # return Job,Machine
    def rule5_4(self):
        # 工件选择规则 5
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_5(self.CTK, self.M_num, self.J_num, self.Jobs, self.NPO, self.J, self.D,
                                          self.CRJ, self.Op_num, self.Processing_time, self.JOB_MAC,
                                          self.transportJS, self.jobShop_Dict)
        # 机器选择规则 4
        Machine = ActionSet.Mac_Selection_4(Job_i, self.Jobs, self.M_num, self.Processing_time)
        # print('This is from rule 5-4:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 5-5
    # J5 M5
    # return Job,Machine
    def rule5_5(self):
        # 工件选择规则 5
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_5(self.CTK, self.M_num, self.J_num, self.Jobs, self.NPO, self.J, self.D,
                                          self.CRJ, self.Op_num, self.Processing_time, self.JOB_MAC,
                                          self.transportJS, self.jobShop_Dict)
        # 机器选择规则 5
        Machine = ActionSet.Mac_Selection_5(Job_i, self.Jobs, self.CTK, self.M_num, self.Processing_time)
        # print('This is from rule 5-5:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 6-1
    # J6 M1
    # return Job,Machine
    def rule6_1(self):
        # 工件选择规则 6
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_6(self.CRJ)
        # 机器选择规则 1
        Machine = ActionSet.Mac_Selection_1(Job_i, self.Jobs, self.Ai, self.CTK, self.Processing_time)
        # print('This is from rule 6-1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 6-2
    # J6 M2
    # return Job,Machine
    def rule6_2(self):
        # 工件选择规则 6
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_6(self.CRJ)
        # 机器选择规则 2
        Machine = ActionSet.Mac_Selection_2(Job_i, self.Jobs, self.CTK, self.Processing_time, self.jobShop_Dict,
                                            self.JOB_MAC, self.transportMAC, self.transport_energy, self.transportJS,
                                            self.Machines)
        # print('This is from rule 6-2:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 6-3
    # J6 M3
    # return Job,Machine
    def rule6_3(self):
        # 工件选择规则 6
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_6(self.CRJ)
        # 机器选择规则 3
        Machine = ActionSet.Mac_Selection_3(Job_i, self.Jobs, self.UR, self.Processing_time)
        # print('This is from rule 6-3:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 6-4
    # J6 M4
    # return Job,Machine
    def rule6_4(self):
        # 工件选择规则 6
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_6(self.CRJ)
        # 机器选择规则 4
        Machine = ActionSet.Mac_Selection_4(Job_i, self.Jobs, self.M_num, self.Processing_time)
        # print('This is from rule 6-4:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 6-5
    # J6 M5
    # return Job,Machine
    def rule6_5(self):
        # 工件选择规则 6
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_6(self.CRJ)
        # 机器选择规则 5
        Machine = ActionSet.Mac_Selection_5(Job_i, self.Jobs, self.CTK, self.M_num, self.Processing_time)
        # print('This is from rule 6-5:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 7-1
    # J7 M1
    # return Job,Machine
    def rule7_1(self):
        # 工件选择规则 7
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_7(self.J_num, self.NPO, self.J, self.D)
        # 机器选择规则 1
        Machine = ActionSet.Mac_Selection_1(Job_i, self.Jobs, self.Ai, self.CTK, self.Processing_time)
        # print('This is from rule 6-1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 7-2
    # J7 M2
    # return Job,Machine
    def rule7_2(self):
        # 工件选择规则 7
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_7(self.J_num, self.NPO, self.J, self.D)
        # 机器选择规则 2
        Machine = ActionSet.Mac_Selection_2(Job_i, self.Jobs, self.CTK, self.Processing_time, self.jobShop_Dict,
                                            self.JOB_MAC, self.transportMAC, self.transport_energy,
                                            self.transportJS,
                                            self.Machines)
        # print('This is from rule 6-2:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 7-3
    # J7 M3
    # return Job,Machine
    def rule7_3(self):
        # 工件选择规则 7
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_7(self.J_num, self.NPO, self.J, self.D)
        # 机器选择规则 3
        Machine = ActionSet.Mac_Selection_3(Job_i, self.Jobs, self.UR, self.Processing_time)
        # print('This is from rule 6-3:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 7-4
    # J7 M4
    # return Job,Machine
    def rule7_4(self):
        # 工件选择规则 7
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_7(self.J_num, self.NPO, self.J, self.D)
        # 机器选择规则 4
        Machine = ActionSet.Mac_Selection_4(Job_i, self.Jobs, self.M_num, self.Processing_time)
        # print('This is from rule 6-4:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 7-5
    # J7 M5
    # return Job,Machine
    def rule7_5(self):
        # 工件选择规则 7
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_7(self.J_num, self.NPO, self.J, self.D)
        # 机器选择规则 5
        Machine = ActionSet.Mac_Selection_5(Job_i, self.Jobs, self.CTK, self.M_num, self.Processing_time)
        # print('This is from rule 6-5:',Machine)
        return Job_i, Machine

    def rule5_6(self):
        # 工件选择规则 5
        ActionSet = ActionMethod()
        Job_i = ActionSet.Job_Selection_5(self.CTK, self.M_num, self.J_num, self.Jobs, self.NPO, self.J, self.D,
                                          self.CRJ, self.Op_num, self.Processing_time, self.JOB_MAC,
                                          self.transportJS, self.jobShop_Dict)
        # 机器选择规则 6
        Machine = ActionSet.Mac_Selection_6(Job_i, self.Jobs, self.M_num, self.Processing_time, self.J_num, self.NPO, self.J)
        # print('This is from rule 1-5:',Machine)
        return Job_i, Machine

    def scheduling(self, action):
        Job, Machine = action[0], action[1]
        O_n = len(self.Jobs[Job].End)  # 第n工序
        # print(Job, Machine,O_n)
        Idle = self.Machines[Machine].idle_time()
        transTime = 0
        try:
            last_ot = max(self.Jobs[Job].End)  # 上道工序加工完成时间
        except:
            last_ot = 0
        try:
            last_mt = max(self.Machines[Machine].End)  # 机器最后完工时间
        except:
            last_mt = 0
        # 运输时间
        if self.last_MAC[Job] == -1:
            transTime = 0
        else:
            if self.transportMAC[self.last_MAC[Job]][Machine] != -1:
                transTime = self.transportMAC[self.last_MAC[Job]][Machine]
            else:
                transTime = self.transportJS[self.jobShop_Dict[self.last_MAC[Job]][0]][self.jobShop_Dict[Machine][0]]
        self.last_MAC[Job] = Machine
        # 开始时间计算方法
        if last_ot >= last_mt:
            Start_time = last_ot + transTime
        else:
            if last_mt > (last_ot + transTime):
                Start_time = last_mt
            else:
                Start_time = last_ot + transTime
        # 工序加工时间
        PT = self.Processing_time[Job][O_n][Machine]
        # 添加运输时间
        for i in range(len(Idle)):  # 计算工序加工开始时间（运输时间）
            if Idle[i][1] - Idle[i][0] > PT:  # 空闲时间大于工序加工时间
                if Idle[i][0] >= Start_time:  # 空闲开始时间大于上道工序加工完成时间
                    Start_time = Idle[i][0]
                    pass
                if Idle[i][0] < Start_time and Idle[i][1] - Start_time > PT:  # 空闲开始时间小于上道工序加工完成时间，空闲结束时间大于加工时间
                    Start_time = Start_time
                    pass
        if Start_time < self.Ai[Job]: Start_time = self.Ai[Job]
        end_time = Start_time + PT  # Start_time include TransTime
        Seq_Job = Job, O_n, Machine, PT, transTime  # 工件i工序j (添加运输时间和空载时间)
        self.Machines[Machine]._add(Start_time, end_time, Job, PT, 0, )
        self.Jobs[Job]._add(Start_time, end_time, Machine, PT, transTime, )
        self._Update(Job, Machine, Seq_Job)

    def Immediate_reward(self, Ta_t, Te_t, Ta_t1, Te_t1, U_t, U_t1, W_t, W_t1, EI_t, EI_t1):
        '''
               :param Ta_t: Tard_a(t)
               :param Te_t: Tard_e(t)
               :param Ta_t1: Tard_a(t+1)
               :param Te_t1: Tard_e(t+1)
               :param U_t: U_ave(t)
               :param U_t1: U_ave(t+1)
               :param W_t: WTard_e(t)
               :param W_t1: WTard_e(t+1)
               :return: reward
        '''
        # 即时奖励（经济指标）
        if Ta_t1 < Ta_t:
            reward_Eco = 1
        elif Ta_t1 > Ta_t:
            reward_Eco = -1
        else:
            if W_t1 < W_t:
                reward_Eco = 1
            elif W_t1 > W_t:
                reward_Eco = -1
            else:
                if Te_t1 < Te_t:
                    reward_Eco = 1
                elif Te_t1 > Te_t:
                    reward_Eco = -1
                else:
                    if U_t1 > U_t:
                        reward_Eco = 1
                    else:
                        if U_t1 > 0.95 * U_t:
                            reward_Eco = 0
                        else:
                            reward_Eco = -1
        # 即时奖励（能耗指标）
        if EI_t > EI_t1:
            reward_En = 1
        elif EI_t < EI_t1:
            reward_En = -1
        else:
            reward_En = 0
        reward = 0.5 * reward_Eco + 0.5 * reward_En
        return reward

    def Cumulative_reward(self, tardiness_t, energy_t, tardiness_t1, energy_t1):
        if tardiness_t == 0:
            reward_Eco = 0
        elif tardiness_t1 < tardiness_t:
            reward_Eco = 1
        elif tardiness_t1 > tardiness_t:
            reward_Eco = -50
        else:
            reward_Eco = -50
        # 能耗奖励
        if energy_t == 0:
            reward_En = 0
        elif energy_t1 < energy_t:
            reward_En = 1
        elif energy_t1 > energy_t:
            reward_En = -50
        else:
            reward_En = -50
        # reward = 100 * (0.5 * reward_Eco + 0.5 * reward_En)
        reward = 0.8 * reward_Eco + 0.2 * reward_En
        return reward

    def mac_based_reward(self, mac_0, mac_1, mac_2, mac_3, mac_4, mac_5, mac_6, mac_7, mac_8, mac_9):
        reward = 0
        reward += -mac_0
        reward += -mac_1
        reward += -mac_2
        reward += -mac_3
        reward += -mac_4
        reward += -mac_5
        reward += -mac_6
        reward += -mac_7
        reward += -mac_8
        reward += -mac_9
        return reward

#  -10 -15

Sit = Situation(J_num, M_num, O_num, J, Processing_time, D, A, Op_num, jobShop_Dict, transportJS, transportMAC)
