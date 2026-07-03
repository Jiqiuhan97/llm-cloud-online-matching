"""
Online Matching Framework Integrating LLMs and the Cloud Model
==============================================================
Implements the proposed framework and three baseline strategies:
  - Proposed   : cloud-model parameters + dynamic reservation threshold
                 + information unreliability penalty + 3En fuzzy early exit
  - FCFS       : First-Come, First-Served
  - Greedy-Det : Deterministic greedy (expectation only)
  - Greedy-TFN : Greedy with triangular fuzzy number defuzzification

Reference:
  Jia et al., "Emergency response oriented to semantically uncertain social
  media texts: An online matching framework integrating large language models
  and the cloud model."

Usage:
  from matching_framework import EventDrivenMatchingSystem

  sys = EventDrivenMatchingSystem('demand.csv', 'supply.csv', model_type='Proposed')
  sys.run_simulation()
  metrics = sys.evaluate_metrics()
"""

import math
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance (km)."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


class EventDrivenMatchingSystem:
    """
    Event-driven online supply-demand matching system.

    Parameters
    ----------
    demand_file : str | pd.DataFrame
        Path to demand CSV, or pre-loaded DataFrame.
    supply_file : str | pd.DataFrame
        Path to supply CSV, or pre-loaded DataFrame.
    model_type : str
        'Proposed' | 'FCFS' | 'Greedy-Det' | 'Greedy-TFN'.
    scarcity_factor : float, default=1.0
        Supply quantity multiplier (for Resource Scarcity experiments).
    """

    def __init__(self, demand_file, supply_file, model_type='Proposed', scarcity_factor=1.0):
        if model_type not in ('Proposed', 'FCFS', 'Greedy-Det', 'Greedy-TFN'):
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model_type = model_type
        self.T_golden = 72.0
        self.epsilon = 1e-5

        self.df_demand = (demand_file if isinstance(demand_file, pd.DataFrame)
                          else pd.read_csv(demand_file)).copy()
        self.df_supply = (supply_file if isinstance(supply_file, pd.DataFrame)
                          else pd.read_csv(supply_file)).copy()
        self.df_supply['Y_jq'] = (self.df_supply['Y_jq'] * scarcity_factor).apply(math.ceil)

        self.I_noise = set(self.df_demand[self.df_demand['He_r,i'] > 1.0].index.tolist())

        all_lons = pd.concat([self.df_demand['lon_i'], self.df_supply['lon_j']])
        all_lats = pd.concat([self.df_demand['lat_i'], self.df_supply['lat_j']])
        self.D_max = haversine(all_lons.min(), all_lats.min(),
                               all_lons.max(), all_lats.max()) or 1.0

        self.allocation_log = []
        self.true_utility_matrix = {}

    # ----------------------------------------------------------------
    # Node parameter extraction
    # ----------------------------------------------------------------
    def _get_node_params(self, row):
        p = row.copy()
        if self.model_type == 'Proposed':
            p['D'], p['R'] = row['Ex_q,i'], row['Ex_r,i']
            p['En_q'], p['En_r'] = row['En_q,i'], row['En_r,i']
            p['He_q'], p['He_r'] = row['He_q,i'], row['He_r,i']
        elif self.model_type == 'FCFS':
            p['D'], p['R'] = row['x,i_FCFS'], row['r,i_FCFS']
            p['En_q'] = p['En_r'] = p['He_q'] = p['He_r'] = 0
        elif self.model_type == 'Greedy-Det':
            p['D'], p['R'] = row['x,i_Det'], row['r,i_Det']
            p['En_q'] = p['En_r'] = p['He_q'] = p['He_r'] = 0
        elif self.model_type == 'Greedy-TFN':
            p['D'], p['R'] = row['x,i_TFN'], row['r,i_TFN']
            p['En_q'] = p['En_r'] = p['He_q'] = p['He_r'] = 0
        p['Initial_D'] = p['D']
        return p

    # ----------------------------------------------------------------
    # Hybrid multi-attribute matching utility  (Eq. 11-16)
    # ----------------------------------------------------------------
    def _decision_utility(self, i_data, j_data, t_m):
        if self.model_type == 'FCFS':
            return -i_data['t_i']

        # Risk urgency
        if self.model_type == 'Proposed':
            R_safe = max(0, (i_data['R'] - 3 * i_data['En_r']) / 100.0)
        else:
            R_safe = i_data['R'] / 100.0
        phi_urgency = (i_data['unit_i'] / 3.0) * R_safe

        # Capacity fitness
        S_jq = j_data['Y_jq']
        if self.model_type == 'Proposed' and i_data['En_q'] > 0:
            phi_fit = (math.exp(-((S_jq - i_data['D']) ** 2)
                                / (2 * i_data['En_q'] ** 2 + self.epsilon))
                       if S_jq <= i_data['D']
                       else i_data['D'] / (S_jq + self.epsilon))
        else:
            phi_fit = (min(1.0, i_data['D'] / (S_jq + self.epsilon))
                       if S_jq > i_data['D'] else 1.0)

        # Logistics cost
        d = haversine(i_data['lon_i'], i_data['lat_i'],
                      j_data['lon_j'], j_data['lat_j'])
        phi_cost = math.exp(-d / self.D_max)

        # Time cost
        T_tol = self.T_golden * (1.0 - i_data['R'] / 100.0 + self.epsilon)
        phi_time = max(0.0, 1.0 - (t_m - i_data['t_i']) / T_tol)
        if phi_time == 0:
            return 0.0

        base_u = 0.4 * phi_urgency + 0.3 * phi_fit + 0.2 * phi_cost + 0.1 * phi_time

        if self.model_type == 'Proposed':
            return base_u * math.exp(-(i_data['He_r'] ** 2))
        return base_u

    # ----------------------------------------------------------------
    # True utility (for offline optimal, always uses full cloud params)
    # ----------------------------------------------------------------
    def _true_utility(self, i_row, j_row, t_m):
        R_safe = max(0, (i_row['Ex_r,i'] - 3 * i_row['En_r,i']) / 100.0)
        phi_urgency = (i_row['unit_i'] / 3.0) * R_safe

        S_jq, Ex, En = j_row['Y_jq'], i_row['Ex_q,i'], i_row['En_q,i']
        phi_fit = (math.exp(-((S_jq - Ex) ** 2) / (2 * En ** 2 + self.epsilon))
                   if En > 0 and S_jq <= Ex
                   else Ex / (S_jq + self.epsilon))

        d = haversine(i_row['lon_i'], i_row['lat_i'],
                      j_row['lon_j'], j_row['lat_j'])
        phi_cost = math.exp(-d / self.D_max)

        T_tol = self.T_golden * (1.0 - i_row['Ex_r,i'] / 100.0 + self.epsilon)
        phi_time = max(0.0, 1.0 - (t_m - i_row['t_i']) / T_tol)
        if phi_time == 0:
            return 0.0

        return (0.4 * phi_urgency + 0.3 * phi_fit + 0.2 * phi_cost + 0.1 * phi_time
                ) * math.exp(-(i_row['He_r,i'] ** 2))

    # ----------------------------------------------------------------
    # Event-driven simulation
    # ----------------------------------------------------------------
    def run_simulation(self):
        for i, r_i in self.df_demand.iterrows():
            for j, r_j in self.df_supply.iterrows():
                if r_i['Q_i'] == r_j['Q_j']:
                    self.true_utility_matrix[(i, j)] = self._true_utility(r_i, r_j, r_j['t_j'])

        events = [(row['t_i'], 'DEMAND', idx, self._get_node_params(row))
                  for idx, row in self.df_demand.iterrows()]
        events += [(row['t_j'], 'SUPPLY', idx, row.copy())
                   for idx, row in self.df_supply.iterrows()]
        events.sort(key=lambda x: x[0])

        I_pool, J_pool = {}, {}

        for t_event, e_type, e_id, e_data in events:
            if e_type == 'DEMAND':
                I_pool[e_id] = e_data
            else:
                t_m = t_event
                J_pool[e_id] = e_data

                while J_pool and I_pool:
                    expired = [i for i, d in I_pool.items()
                               if (t_m - d['t_i']) >= self.T_golden
                               * (1.0 - d['R'] / 100.0 + self.epsilon)]
                    for idx in expired:
                        del I_pool[idx]
                    if not I_pool:
                        break

                    valid = []
                    for j_id, jd in J_pool.items():
                        for i_id, id_ in I_pool.items():
                            if id_['Q_i'] == jd['Q_j']:
                                u = self._decision_utility(id_, jd, t_m)
                                if u > 0 or self.model_type == 'FCFS':
                                    valid.append({'i': i_id, 'j': j_id, 'U': u, 'q': jd['Q_j']})
                    if not valid:
                        break

                    best = max(valid, key=lambda x: x['U'])

                    # Dynamic reservation threshold
                    if self.model_type == 'Proposed':
                        q_pairs = [x for x in valid if x['q'] == best['q']]
                        tD = sum(d['D'] for d in I_pool.values() if d['Q_i'] == best['q'])
                        tS = sum(j['Y_jq'] for j in J_pool.values() if j['Q_j'] == best['q'])
                        rho = min(1.0, tS / (tD + self.epsilon))
                        tau = max(0.1, np.percentile([x['U'] for x in q_pairs],
                                                     (1.0 - rho) * 100))
                    elif self.model_type == 'FCFS':
                        tau = -float('inf')
                    else:
                        tau = 0.0

                    if best['U'] < tau:
                        break

                    id_, jd = I_pool[best['i']], J_pool[best['j']]
                    transfer = min(id_['D'], jd['Y_jq'])
                    jd['Y_jq'] -= transfer
                    id_['D'] -= transfer

                    self.allocation_log.append({
                        'i': best['i'], 'j': best['j'], 'amount': transfer,
                        'True_U': self.true_utility_matrix.get((best['i'], best['j']), 0),
                        't_match': t_m, 't_i': id_['t_i']
                    })

                    if jd['Y_jq'] <= 0:
                        del J_pool[best['j']]
                    if self.model_type == 'Proposed':
                        if id_['D'] <= 3 * id_['En_q']:
                            del I_pool[best['i']]
                    elif id_['D'] <= 0:
                        del I_pool[best['i']]

    # ----------------------------------------------------------------
    # Offline optimal
    # ----------------------------------------------------------------
    def _solve_hindsight_opt(self):
        ni, nj = len(self.df_demand), len(self.df_supply)
        C = np.zeros((ni, nj))
        for i in range(ni):
            for j in range(nj):
                if self.df_demand.iloc[i]['Q_i'] == self.df_supply.iloc[j]['Q_j']:
                    C[i, j] = self.true_utility_matrix.get((i, j), 0)

        c = -C.flatten()
        A_s = np.zeros((nj, ni * nj))
        for j in range(nj):
            A_s[j, j::nj] = 1.0
        A_d = np.zeros((ni, ni * nj))
        for i in range(ni):
            A_d[i, i * nj:(i + 1) * nj] = 1.0

        res = linprog(c, A_ub=np.vstack([A_s, A_d]),
                      b_ub=np.concatenate([self.df_supply['Y_jq'].values,
                                           self.df_demand['Ex_q,i'].values]),
                      bounds=[(0, None)] * (ni * nj), method='highs')
        return -res.fun if res.success else 0.0

    # ----------------------------------------------------------------
    # Metrics  (Eq. 20-25)
    # ----------------------------------------------------------------
    def evaluate_metrics(self, base_opt_utility=None, scale_multiplier=1):
        df_log = pd.DataFrame(self.allocation_log)

        rsr_list = []
        for i_id, row in self.df_demand.iterrows():
            if i_id in self.I_noise:
                continue
            params = self._get_node_params(row)
            ex_q = params['Initial_D']
            if df_log.empty or i_id not in df_log['i'].values:
                rsr_i = 0.0
            else:
                rc = df_log[df_log['i'] == i_id]['amount'].sum()
                rsr_i = min(1.0, rc / (ex_q + self.epsilon))
                if self.model_type == 'Proposed' and (ex_q - rc) <= 3 * params['En_q']:
                    rsr_i = 1.0
            rsr_list.append(rsr_i)

        Eff_RSR = np.mean(rsr_list) if rsr_list else 0.0
        n = len(rsr_list)
        Eff_Gini = (sum(abs(x - y) for x in rsr_list for y in rsr_list)
                     / (2.0 * n ** 2 * Eff_RSR)) if Eff_RSR > 0 and n > 0 else 1.0

        AWT = ((df_log['t_match'] - df_log['t_i']).mean()
               if not df_log.empty else 0.0)

        if self.I_noise:
            mn = (set(df_log['i'].unique()).intersection(self.I_noise)
                  if not df_log.empty else set())
            NIR = 1.0 - len(mn) / len(self.I_noise)
        else:
            NIR = 1.0

        online_u = (df_log['amount'] * df_log['True_U']).sum() if not df_log.empty else 0.0
        offline_opt = (base_opt_utility * scale_multiplier
                       if base_opt_utility is not None
                       else self._solve_hindsight_opt())
        CR = online_u / offline_opt if offline_opt > 0 else 0.0

        return {'CR': round(CR, 4), 'Eff_RSR': round(Eff_RSR, 4),
                'AWT (h)': round(AWT, 2), 'Eff_Gini': round(Eff_Gini, 4),
                'NIR': round(NIR, 4), 'Model': self.model_type}
