import numpy as np
import random


class ActionMethod:
    # Tard_Job不为空，则选择集合中EDT*Wi最大的工件
    # Tard_Job为空, 选择平均松弛时间最小的工件
    def Job_Selection_1(self, CTK, M_num, J_num, Jobs, NPO, J, D, Processing_time):
        # T_cur:所有机器中最后一个工序的平均完工时间
        T_cur = sum(CTK) / M_num
        # Tard_Job:不能按期完成的工件
        Tard_Job = []
        for i in range(J_num):
            if not Jobs[i].End:
                if NPO[i] < J[i][0] and T_cur >= D[i]:
                    Tard_Job.append(i)
            else:
                if NPO[i] < J[i][0] and max(T_cur, Jobs[i].End[-1]) >= D[i]:
                    Tard_Job.append(i)
        # 未完成工件集合
        UC_Job = [j for j in range(J_num) if NPO[j] < J[j][0]]
        # 最小松弛时间
        slack_lst = []
        if not Tard_Job:
            # 最小松弛时间的未完成工件
            for i in UC_Job:
                if not Jobs[i].End:
                    slack_time = (D[i] - T_cur) / (J[i][0] - NPO[i])
                    slack_lst.append(slack_time)
                else:
                    slack_time = (D[i] - max(T_cur, Jobs[i].End[-1])) / (J[i][0] - NPO[i])
                    slack_lst.append(slack_time)
            Job_i = UC_Job[np.argmin(slack_lst)]
        else:
            # EDT*Wi最大的延迟工件
            EDT_Wi_lst = []
            for i in Tard_Job:
                Process_left = []
                # Trans_left = []
                for j in range(NPO[i], J[i][0]):
                    process_ijk = [k for k in Processing_time[i][j] if k != -1]
                    # trans_ijk = [k for k in self.Jobs[i].Transport[]]
                    Process_left.append(sum(process_ijk) / len(process_ijk))
                    # Trans_left.append(sum(trans_ijk) / len(trans_ijk))
                try:
                    EDT_Wi = (max(T_cur, Jobs[i].End[-1]) + sum(Process_left) - D[i]) * J[i][1]
                except:
                    EDT_Wi = (T_cur + sum(Process_left) - D[i]) * J[i][1]
                EDT_Wi_lst.append(EDT_Wi)  #
            Job_i = Tard_Job[np.argmax(EDT_Wi_lst)]
        return Job_i

    # Tard_Job不为空，则选择集合中EDT*Wi最大的工件
    # Tard_Job为空, 选择临界比值最小的工件(剩余加工时间临界比=工件i允许加工时长/工件i剩余所需加工时间)
    def Job_Selection_2(self, CTK, M_num, J_num, Jobs, NPO, J, D, Op_num, Processing_time, JOB_MAC, transportJS, jobShop_Dict):
        # T_cur:所有机器中最后一个工序的平均完工时间
        T_cur = sum(CTK) / M_num
        # Tard_Job:不能按期完成的工件
        Tard_Job = []
        for i in range(J_num):
            if not Jobs[i].End:
                if NPO[i] < J[i][0] and T_cur >= D[i]:
                    Tard_Job.append(i)
            else:
                if NPO[i] < J[i][0] and max(T_cur, Jobs[i].End[-1]) >= D[i]:
                    Tard_Job.append(i)
        # 未完成工件集合
        UC_Job = [j for j in range(J_num) if NPO[j] < J[j][0]]
        if not Tard_Job:
            Ti_ave = [0 for i in range(J_num)]  # 工件i剩余所需加工时间 运输时间 + 加工时间
            for i in UC_Job:
                Ti_pro = 0
                Ti_tra = 0
                # 加工时间
                O_n = len(Jobs[i].End)
                for j in range(O_n, Op_num[i]):
                    Tij_ava = [k for k in Processing_time[i][j] if k != -1]  # 可用机器加工时间
                    Ti_pro += sum(Tij_ava) / len(Tij_ava)
                if not JOB_MAC:
                    Ti_tra += 0
                else:
                    Ti_tra += (Op_num[i] - O_n) * (sum(transportJS[jobShop_Dict[JOB_MAC[-1][2]][0]])
                                                   / (len(transportJS[jobShop_Dict[JOB_MAC[-1][2]][0]]) - 1))
                Ti_ave[i] = (Ti_pro + Ti_tra)
            # 最小临界比值的未完成工件
            UC_CRi = []
            for i in UC_Job:
                if not Jobs[i].End:
                    UC_CRi.append((D[i] - T_cur) / Ti_ave[i])
                else:
                    UC_CRi.append((D[i] - max(T_cur, Jobs[i].End[-1])) / Ti_ave[i])
            # if UC_CRi
            Job_i = UC_Job[np.argmin(UC_CRi)]
        else:
            # EDT*Wi最大的延迟工件
            EDT_Wi_lst = []
            for i in Tard_Job:
                Process_left = []
                # Trans_left = []
                for j in range(NPO[i], J[i][0]):
                    process_ijk = [k for k in Processing_time[i][j] if k != -1]
                    # trans_ijk = [k for k in self.Jobs[i].Transport[]]
                    Process_left.append(sum(process_ijk) / len(process_ijk))
                    # Trans_left.append(sum(trans_ijk) / len(trans_ijk))
                try:
                    EDT_Wi = (max(T_cur, Jobs[i].End[-1]) + sum(Process_left) - D[i]) * J[i][1]
                except:
                    EDT_Wi = (T_cur + sum(Process_left) - D[i]) * J[i][1]
                EDT_Wi_lst.append(EDT_Wi)  #
            Job_i = Tard_Job[np.argmax(EDT_Wi_lst)]
        return Job_i

    # 选择EDT * Wi最大的作为下一个调度工序
    def Job_Selection_3(self, CTK, M_num, J_num, NPO, J, D, Processing_time, Jobs):
        # T_cur:所有机器中最后一个工序的平均完工时间
        T_cur = sum(CTK) / M_num
        UC_Job = [j for j in range(J_num) if NPO[j] < J[j][0]]  # 未完成工件
        # EDT*Wi最大的延迟工件
        EDT_Wi_lst = []
        for i in UC_Job:
            Process_left = []
            # Trans_left = []
            for j in range(NPO[i], J[i][0]):
                process_ijk = [k for k in Processing_time[i][j] if k != -1]
                # trans_ijk = [k for k in self.Jobs[i].Transport[]]
                Process_left.append(sum(process_ijk) / len(process_ijk))
                # Trans_left.append(sum(trans_ijk) / len(trans_ijk))
            try:
                EDT_Wi = (max(T_cur, Jobs[i].End[-1]) + sum(Process_left) - D[i]) * J[i][1]
            except:
                EDT_Wi = (T_cur + sum(Process_left) - D[i]) * J[i][1]
            EDT_Wi_lst.append(EDT_Wi)
        Job_i = UC_Job[np.argmax(EDT_Wi_lst)]
        return Job_i

    # 随机选一个未完成的工件
    def Job_Selection_4(self, J_num, NPO, J):
        UC_Job = [j for j in range(J_num) if NPO[j] < J[j][0]]  # 未完成工件
        Job_i = random.choice(UC_Job)
        return Job_i

    # Tard_Job不为空, 选择工件延迟工序临界比与剩余加工时间的乘积最大的工件
    # Tard_Job为空, 选择工件完成率 * 剩余加工时间最小的工件
    def Job_Selection_5(self, CTK, M_num, J_num, Jobs, NPO, J, D, CRJ, Op_num, Processing_time, JOB_MAC, transportJS, jobShop_Dict):
        # T_cur:所有机器中最后一个工序的平均完工时间
        T_cur = sum(CTK) / M_num
        # Tard_Job:不能按期完成的工件
        Tard_Job = []
        for i in range(J_num):
            if not Jobs[i].End:
                if NPO[i] < J[i][0] and T_cur >= D[i]:
                    Tard_Job.append(i)
            else:
                if NPO[i] < J[i][0] and max(T_cur, Jobs[i].End[-1]) >= D[i]:
                    Tard_Job.append(i)
        UC_Job = [j for j in range(J_num) if NPO[j] < J[j][0]]
        if not Tard_Job:
            uc_lst = []
            for ii in UC_Job:
                if not Jobs[ii].End:
                    uc_lst.append(CRJ[ii] * (D[ii] - T_cur))
                else:
                    uc_lst.append(CRJ[ii] * (D[ii] - max(T_cur, Jobs[ii].End[-1])))
            Job_i = UC_Job[np.argmin(uc_lst)]
        else:
            Wi_tardiness = []  # 加权延迟临界比
            Ti_ave = [0 for i in range(J_num)]  # 工件i剩余所需加工时间 运输时间 + 加工时间
            Total_tard = 0  # 所有延迟工件未完成工序
            single_tard = [0 for i in range(J_num)]  # 延迟工件未完成工序数量列表
            for i in Tard_Job:
                Ti_pro = 0
                Ti_tra = 0
                temp = J[i][0] - NPO[i]
                Total_tard += temp
                single_tard[i] = temp
                O_n = len(Jobs[i].End)
                for j in range(O_n, Op_num[i]):
                    Tij_ava = [k for k in Processing_time[i][j] if k != -1]  # 可用机器加工时间
                    Ti_pro += sum(Tij_ava) / len(Tij_ava)
                if not JOB_MAC:
                    Ti_tra += 0
                else:
                    Ti_tra += (Op_num[i] - O_n) * (sum(transportJS[jobShop_Dict[JOB_MAC[-1][2]][0]]) / (
                            len(transportJS[jobShop_Dict[JOB_MAC[-1][2]][0]]) - 1))
                Ti_ave[i] = (Ti_pro + Ti_tra)
            for jobi in Tard_Job:
                Wi_tardiness.append((single_tard[jobi] / Total_tard) * Ti_ave[jobi] * J[jobi][1])
            Job_i = Tard_Job[np.argmax(Wi_tardiness)]
        return Job_i

    # 工件完成率最低优先
    def Job_Selection_6(self, CRJ):
        complete_rate = []
        min_list = []
        for i in range(len(CRJ)):
            complete_rate.append(CRJ[i])
        min_value = min(complete_rate)
        for j in range(len(CRJ)):
            if complete_rate[j] == min_value:
                min_list.append(j)
        Job_i = random.choice(min_list)
        return Job_i

    # 最早截止日期优先
    def Job_Selection_7(self, J_num, NPO, J, D):
        UC_Job = [j for j in range(J_num) if NPO[j] < J[j][0]]  # 未完成工件
        D_list = []
        for i in UC_Job:
            D_list.append(D[i])
        Job_i = UC_Job[np.argmin(D_list)]
        return Job_i

    # 最早可用设备
    def Mac_Selection_1(self, Job_i, Jobs, Ai, CTK, Processing_time):
        try:
            C_ij = max(Jobs[Job_i].End)  # 工件i的结束时间
        except:
            C_ij = Ai[Job_i]
        A_ij = Ai[Job_i]  # 工件i的arrival time
        # print(A_ij)
        On = len(Jobs[Job_i].End)  # 工序序号/索引
        Mk = []
        for i in range(len(CTK)):
            if Processing_time[Job_i][On][i] != -1:
                C_ij += Processing_time[Job_i][On][i]
                Mk.append(max(C_ij, A_ij, CTK[i]))
            else:
                Mk.append(9999)
        # print('This is from rule 1-1:',Mk)
        Machine = np.argmin(Mk)
        return Machine

    # 能耗代价最少的可用设备
    def Mac_Selection_2(self, Job_i, Jobs, CTK, Processing_time, jobShop_Dict, JOB_MAC, transportMAC,
                        transport_energy, transportJS, Machines):
        TE_process = []
        TE_trans = []
        TE_idle = []
        On = len(Jobs[Job_i].End)  # 工序序号/索引
        for i in range(len(CTK)):
            if Processing_time[Job_i][On][i] != -1:
                # 加工能耗
                TE_pro = Processing_time[Job_i][On][i] * jobShop_Dict[i][1]
                TE_process.append(TE_pro)
                # 运输能耗
                if not JOB_MAC:
                    TE_trans.append(0)
                elif JOB_MAC[-1][2] == i:
                    TE_trans.append(0)
                else:
                    if transportMAC[JOB_MAC[-1][2]][i] == -1:
                        TE_tra = transportMAC[JOB_MAC[-1][2]][i] * transport_energy
                        TE_trans.append(TE_tra)
                    else:
                        TE_tra = transportJS[jobShop_Dict[JOB_MAC[-1][2]][0]][jobShop_Dict[i][0]] * transport_energy
                        TE_trans.append(TE_tra)
                # 空载能耗
                try:
                    last_ot = max(Jobs[Job_i].End)  # 上道工序加工完成时间
                except:
                    last_ot = 0
                try:
                    last_mt = max(Machines[i].End)  # 机器最后完工时间
                except:
                    last_mt = 0
                # 运输时间
                if not JOB_MAC:
                    transTime = 0
                elif JOB_MAC[-1][2] == i:
                    transTime = 0
                else:
                    if transportMAC[JOB_MAC[-1][2]][i] != -1:
                        transTime = transportMAC[JOB_MAC[-1][2]][i]
                    else:
                        transTime = transportJS[jobShop_Dict[JOB_MAC[-1][2]][0]][jobShop_Dict[i][0]]
                # 开始时间计算方法
                if last_ot >= last_mt:
                    Start_time = last_ot + transTime
                else:
                    if last_mt > (last_ot + transTime):
                        Start_time = last_mt
                    else:
                        Start_time = last_ot + transTime
                if not Machines[i].End:
                    TE_idl = Start_time * jobShop_Dict[i][2]
                    TE_idle.append(TE_idl)
                else:
                    if Start_time > Machines[i].End[-1]:
                        TE_idl = (Start_time - Machines[i].End[-1]) * jobShop_Dict[i][2]
                        TE_idle.append(TE_idl)
                    else:
                        TE_idle.append(0)
            else:
                TE_process.append(9999)
                TE_trans.append(9999)
                TE_idle.append(9999)
        TE_ij = [TE_p + TE_t + TE_i for TE_p, TE_t, TE_i in zip(TE_process, TE_trans, TE_idle)]
        # print('This is from rule 1-2:',TE_ij)
        Machine = np.argmin(TE_ij)
        return Machine

    # 设备利用率最低的可用设备
    def Mac_Selection_3(self, Job_i, Jobs, UR, Processing_time):
        On = len(Jobs[Job_i].End)  # 工序序号/索引
        # print(self.UR)
        UR_copy = []
        for ur in range(len(UR)):
            UR_copy.append(UR[ur])
        for i in range(len(UR_copy)):
            if Processing_time[Job_i][On][i] == -1:
                UR_copy[i] = 9999
        # print(UR_copy)
        Machine = np.argmin(UR_copy)
        return Machine

    # 最短处理时间的可用设备
    def Mac_Selection_4(self, Job_i, Jobs, M_num, Processing_time):
        On = len(Jobs[Job_i].End)  # 工序序号/索引
        process_shortest = [-1 for i in range(M_num)]
        for i in range(M_num):
            if Processing_time[Job_i][On][i] == -1:
                process_shortest[i] = 9999
            else:
                process_shortest[i] = Processing_time[Job_i][On][i]
        Machine = np.argmin(process_shortest)
        return Machine

    # 最后加工时间最短的可用设备
    def Mac_Selection_5(self, Job_i, Jobs, CTK, M_num, Processing_time):
        On = len(Jobs[Job_i].End)  # 工序序号/索引
        complete_time = []
        available_mac = [99999 for j in range(M_num)]
        for i in range(len(CTK)):
            complete_time.append(CTK[i])
        for j in range(len(CTK)):
            if Processing_time[Job_i][On][j] != -1:
                available_mac[j] = complete_time[j]
        Machine = np.argmin(available_mac)
        return Machine

    # 加工次数的可用设备
    def Mac_Selection_6(self, Job_i, Jobs, M_num, Processing_time, J_num, NPO, J):
        On = len(Jobs[Job_i].End)  # 工序序号/索引
        available_mac = []
        for i in range(M_num):
            if Processing_time[Job_i][On][i] != -1:
                available_mac.append(i)
        UC_Job = [j for j in range(J_num) if NPO[j] < J[j][0]]  # 未完成工件
        number_of_process = [0 for j in range(len(available_mac))]
        for i in UC_Job:
            pre_On = NPO[i]
            for j in available_mac:
                if Processing_time[i][pre_On][j] != -1:
                    for k in range(len(available_mac)):
                        if j == available_mac[k]:
                            number_of_process[k] += 1
        Machine = available_mac[np.argmin(number_of_process)]
        return Machine
