import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
for r in csv.DictReader(open(os.path.join(HERE, 'sensor_compare.csv'))):
    f = lambda k, s=1: ('%.2f' % (float(r[k]) * s)) if r.get(k) not in ('', None) else '-'
    print(f"{r['variant']:11s} {r['name']:8s} <1 {f('err_lt1',100):>6s}% <3 {f('err_lt3',100):>6s}% p50 {f('err_p50'):>6s} "
          f"p90 {f('err_p90'):>6s} rmse r/p/y {f('rmse_roll')}/{f('rmse_pitch')}/{f('rmse_yaw')} ANEES {f('anees')} "
          f"ANIS {f('anis')} (dof {r['dof_meas']}) 3sig {f('in3sig',100)}% bg {r['bg_final_dps']} fit {r['boresight_fit_rpy_deg']}")
