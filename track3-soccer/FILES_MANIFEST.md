# Track3 文件清单（2026-08-06 03:10 CST 实测，路径全部真实存在）

## 永久目录（NFS 持久盘 /persistent，实例重建不丢）
- 根: `/workspace/persistent/track3/`（SPEC §1.2 指定）
- 脚本: `scripts/run_task9_p1.py`（rsl_rl 5.4.2 兼容版，lr=1.5e-4）
- 配置: `configs/hierarchical_agent.yaml`（lr=1.5e-4 锁定，禁调大）
- checkpoint: `runs/task9_p1/model_*.pt`（训练实时同步；阶段末另存 `models/task9_p1.pt`）
- 训练日志: `training_logs/task9_p1.log`（每阶段末同步最新）
- 基线: `reports/task9_baseline/`（3v3_final_stdout: fallen=300/disp=0.78m/FAILED；v2: 倒地球飞）
- 准则: `SPEC.md`（Task-9-v2 唯一准则）+ `MAIN_CONTROLLER_NOTE.md`

## 远端工作区（/workspace 本地盘，实例重建会丢，勿依赖）
- 训练: `run_task9_p1.py --max_iterations 240 --num_envs 2048 --phase A`（setsid nohup 已脱离）
- 运行中产物: `runs/task9_p1/`、`training_logs/task9_p1.log`、`demos/exp/task9_p1_result.json`
- 3v3 演示: `demo_artifacts/`（match_3v3_final.mp4/png、3v3_final_stdout.txt 等，保护勿动）
- 单机射门保底（PASSED）: `demos/exp/match_1v1_shoot_20260805.mp4` + result json

## 本地仓库（Radeon-hackathon-2026-07，branch codex/track3-final-acceptance）
- `track3-soccer/run_task9_p1.py`（补丁版，commit e754605）
- `track3-soccer/reports/task9_baseline/`（本地基线备份）
- `track3-soccer/SPEC.md`（同步源；track3 敏感文件已 gitignore: launch_task9.py / MAIN_AGENT_HANDOFF.md）

## 已确认不存在（交接文档死路径，勿再引用）
- `/workspace/radeon-repo/`（旧实例残留）
- `/workspace/train_1v1.py`、`/workspace/run_curriculum.sh`（均不存在，B 阶段需新建 1v1 训练脚本）
- 旧隧道 `minimize-orders-excel-saving.trycloudflare.com`（DNS 已注销，不可恢复）
- Vercel 部署（CLI 卡 npm dist-tags、token 403，已弃用）

## 10h 窗口关键时间（t0=02:37 CST）
- 2h 检查点 04:37 / 4h 门禁 06:37 / 8h 冻结 10:37 / 10h 终检 12:37
- 止损: fallen per_ep>80 → fall_penalty=-18 + hl_clip=0.5；不动 → approach_ball=7
- 禁 3v3/coop 训练、禁跑 run_12h_schedule.sh（含 coop 段）
