# MaskPepDesigner 命名迁移与代码审查（2026-09-07）

本次完成本地命名迁移及明确的接口/数据组装错误修复。审查重点是当前 v6
训练、数据处理、推理和打包链路，并非对所有上游 Uni-Fold、SSbuilder 和分析
工具的完整审计。没有运行远程任务，也没有用真实 checkpoint/GPU 验证数值结果。

## 迁移契约

| 对象 | 新名称 | 兼容方式 |
| --- | --- | --- |
| Python 主包 | `MPDesigner` | `diffold.*` 模块转发到同一实现 |
| 当前神经网络 | `MPDesigner.nn.mpdesigner.MPDesigner` | `diffold.nn.dfv6.Dfv6`、`MPDesigner.nn.dfv6.Dfv6` 保留别名 |
| 训练模型/任务/损失 | `MPDesignerModel` / `MPDesignerTask` / `MPDesignerLoss` | 保留 `diffold` 注册，使用独立兼容子类避免注册类名冲突 |
| Uni-Core 架构注册 | `MPDesigner_v6` | 保留 `diffold_v6` |
| Dataset / 辅助头 / Evoformer | `MPDesignerDataset` / `MPDesignerAuxHeads` / `MPDesignerEvoformer*` | 保留原类名别名 |
| 推理入口 | `run_MPDesigner.py` / `_ss.py` / `_ori.py` | 原 `run_diffold*.py` 转发 |
| 训练入口 | `train_MPDesigner*.sh` / `finetune_MPDesigner*.sh` | 原脚本转发；`finetune_ufcpdiff.sh` 转到 `finetune_MPDesigner_peptide.sh` |
| 安装分发包名 | `MaskPepDesigner` | 保留 `unifold` Python 包，但不再以 Uni-Fold 名义发布此分发包 |

`MPDesigner_v6` 是训练框架的注册名，不能代替具体配置。`--model-name` /
`--model` 必须使用与 checkpoint 一致的版本及全部修饰符，例如原来 `v65`
配置可继续用原名，或加上 `MPDesigner_` 前缀。`denovo`、`v62`–`v66` 等
配置版本没有被合并或重解释。缺少配置时明确报错，不擅自选择新版本。

网络层结构、参数属性名、state_dict 的参数键及张量尺寸没有因改名而调整。
这通过当前网络文件的 AST 去除命名变化后与 HEAD 比较得到；不是实际
checkpoint 加载成功的证明。完整对象 pickle 及外部代码兼容仍需在训练环境验证。

旧版 `nn/diffold.py`、`losses/diffold.py` 和原开发测试保留历史身份，没有把
前 v6 网络伪装成当前 MPDesigner。`unifold/` 上游代码及原版权、许可证未修改。
新文档明确说明 MaskPepDesigner → Diffold → DP Technology Uni-Fold 的开发关系。
仓库原有已跟踪的 `__pycache__` 未删除；新增忽略规则防止继续引入。

## 已修复

1. **重复引用 PDB 残基污染样本**：`MPDesigner/inference/utils.py:280` 原来直接
   修改 `pdb_ref` 中的字典，同一残基用于多个输出链时，后一次修改会覆盖前一次
   chain ID、生成掩码和 hotspot。现在每次深拷贝；测试覆盖同一残基的不同链/掩码。
2. **无 contig 时元组长度不匹配**：`MPDesigner/inference/utils.py:313` 的调用
   从三元素补齐为五元素。仅修复样本组装；默认 spatial crop 的无 hotspot
   限制仍存在，见下文。
3. **超时占位样本拼接失败**：`MPDesigner/dataset.py:190` 原来对 dummy 读取
   `aatype`，触发 KeyError。现在混合 batch 过滤 dummy，全 dummy batch 保留
   既有 loss 分支所需占位格式。分布式全 dummy 的梯度同步仍未验证。
4. **训练配置默认值不可用**：`MPDesigner/model.py:49` 原来可能得到 None 或
   不受配置解析器支持的 `diffold_v6`。现在要求显式配置并提供明确报错。
5. **指标归约修改输入**：`MPDesigner/loss.py:46` 原来的 `pop('sample_size')`
   会使同一日志再次归约被过滤。现在只读访问，测试确认重复归约得到相同结果。
6. **零步/零副本任务**：`MPDesigner/inference/job.py:36` 新增正整数检查，避免
   零步后访问未赋值输出或零副本任务无声不产生结果。
7. **启动脚本错误**：修正训练/demo 创建目录参数、demo 不存在的 `diffold_v4`
   注册名、finetune 的固定 TensorBoard 目录。添加参数数量检查、临时目录 EXIT
   清理及训练退出码传递；原脚本最后 `rm` 成功会掩盖训练失败。
8. **文档/打包错误**：README 改成真实 CLI 参数和五参数训练示例；纠正
   inference 的 `__init___.py` 拼写，补 losses 包标记及 spline package_data，
   验证 wheel 包含这些文件。pip 依赖声明仍不是完整 GPU 环境锁文件。

## 尚未修复的审查发现（按影响排序）

以下均有源码依据；涉及数值行为的修复应独立变更并与论文使用的 checkpoint
和基准输入比较，防止命名迁移同时改变实验定义。

### P1：固定区域没有落实到最终 PDB

位置：`run_MPDesigner.py:177,213,241`；`MPDesigner/diffusion/transition_v6.py`
的 `denoise` / `denoise_fwd`。`_ss`、`_ori` 入口也采用原始网络输出保存。

去噪步骤使用 `seq_gen_mask`、`frame_gen_mask` 保持固定输入，返回 `s_s/f_s`；
最终保存却使用 `sh_0/fh_0`。因此对 receptor/motif 要求序列或结构固定时，
输出没有得到同样的掩码保证，即使中间扩散状态已经保持固定区域。
建议明确最终输出契约，保存最后约束后的状态，并逐残基比较固定区序列和坐标。

### P1：预测预处理无条件加噪声，种子不能完全控制复现

位置：`MPDesigner/data/process.py:35`、`MPDesigner/data/data_ops.py:29,48`。

训练、评估、预测共用流程，都会调用默认 `add_noise=True` 的预处理；
噪声来自 `torch.randn_like`，而流程种子上下文控制的是 NumPy。相同 Job seed
不能单独保证这部分噪声相同，参考 receptor 坐标也被扰动。建议将训练增强
设为配置项，预测默认关闭；如果保留，显式管理 Torch generator，并检查重复
运行的输入特征及最终坐标。此轮未改变噪声分布或开启条件。

### P1：“恢复原始坐标”没有恢复前处理移除的平移

位置：`MPDesigner/data/data_ops.py:125,169`；`run_MPDesigner.py:123,231`。

`make_initial_frames` 已将参考坐标中心化且覆盖 `all_atom_positions`，没有输出
移除的原始中心。入口随后从处理后的 batch 计算 `pocket_center`，最后只加回
这个处理后中心，不能恢复更早移除的平移。对非原点输入，这不足以兑现日志中
“back to original coordinates”的表述。建议在 crop/augmentation 阶段记录完整
刚体变换，所有输出使用同一逆变换；用已知平移的同一 PDB 做回归验证。
此外首次移动 frames 后没有立即重算 `diffused_quats`；`fif` 配置应专项检查。

### P1：没有显式 hotspot 的合法输入在默认 crop 流程失败

位置：`MPDesigner/inference/utils.py:307`、`MPDesigner/data/data_ops.py:538`。

样本总会包含 `hpmask`，没有 `{...}` 时为全零；spatial crop 只判断是否为 None，
随后对全零直接 ValueError。这发生在“小于 crop_size 不裁切”的早退之前，因此
旧文档普通 `A1-153`、无 contig 输入甚至短输入也不能依赖大小绕过问题。
建议明确支持的模式：有参考 peptide 时可选择界面推断；纯 de novo 缺 hotspot
时应在任务输入层明确拒绝或要求口袋定义，不能随意选中心。新文档已要求显式
hotspot，但未改变原裁切行为。

### P2：无结构肽的保留只覆盖一个裁切分支

位置：`MPDesigner/data/data_ops.py:567,595`。

现有 `min_distances[peptide_indices] = -inf` 已保留，防止回退历史 de novo 修复。
但在 `ca_mask.sum() <= crop_size` 且总长超限时，另一个分支优先保留全部有结构
残基，再随机选无结构残基：可能只保留部分待生成 peptide。若 peptide 本身超过
crop_size，也没有明确完整性验证。建议所有分支共用“保留生成区域，再分配
receptor 预算”的约束，并对不可能满足的长度提前报错。

### P2：链身份假设限制泛化输入

位置：`MPDesigner/data/data_ops.py:274,534`、`MPDesigner/inference/utils.py:211`。

多处把内部 chain 1 当 receptor、chain 2 当 peptide，而不是统一依据 ligmask
区分。反向链序或多链 receptor 的裁切、中心和训练掩码可能不符合用户预期。
新文档显式要求 receptor 在前、peptide 在后；长期应定义统一角色字段，并覆盖
多链 receptor 的回归样例。

### P2：支持的配置模式与推理入口输出假设不一致

位置：`MPDesigner/config_v6.py:417`、`run_MPDesigner.py:177`。

`seq_only` 关闭 decoder 和辅助头，`struct_only` 关闭序列头；推理入口仍无条件
读取 `frames`、`aalogits`、`plddt`。选择这些修饰符时可能出现 KeyError。
应在 CLI 初始化时拒绝不支持的模式，或按任务实现相应输出流程。

### P2：历史开发测试不能作为当前网络验收

位置：`MPDesigner/nn/diffold.py:21`、`MPDesigner/test/test_diffold.py`。

旧网络导入已不存在的 `DiffoldAuxHeads`，旧测试还使用前 v6 输入/配置接口并在
模块顶层运行。不能只把这个类名替换成新头就假设架构兼容。旧测试保留供溯源，
并从分发包中排除；本轮新增的 `tests/` 是独立、可重复执行的轻量回归集合。

### 需要环境验证的风险

- 全 dummy loss 是独立的 requires_grad 标量，没有通过模型参数构图；不同 DDP
  rank 有效样本数不同时的同步和归约行为尚未验证。collater 修复不证明这一场景安全。
- 三个推理入口的 checkpoint loader 假定 `ema.params` / `model` 格式并无条件
  去掉每个参数键的第一段。裸网络 state_dict、不同保存格式及跨设备加载需单独验证。
- 当前依赖声明没有覆盖完整 Uni-Core/GPU/后处理环境，也没有锁版本。新 wheel
  内容检查只证明包内容齐全到本轮检查范围，不能证明干净环境安装后直接运行。
- 已有 `utils_cutpock.py` 等历史副本与主入口可能继续分叉，后续应指定维护入口后归档。

## 本次验证与复核步骤

- `python -B -m unittest discover -s tests -v`：12 项通过。使用 NumPy/标准库，
  部分函数通过 AST 提取，导入检查对训练依赖使用 stub；不替代真实框架集成测试。
- 102 个相关 Python 文件通过 AST 解析；根目录 shell 脚本全部通过 `bash -n`。
- 当前网络 `nn/mpdesigner.py` 与改名前 `nn/dfv6.py` 的 AST 在归一化命名和去除
  兼容别名后完全一致，验证本轮没有改网络计算逻辑。
- `setup.py bdist_wheel` 成功；检查 inference/losses、新旧包和 spline 资源存在，
  历史测试及 pycache 不进入 wheel。本轮生成的仓库 build/egg-info 已清理。
- `git diff --check` 通过；原有未跟踪 `AGENTS.md` 没有修改。

建议在用于论文实验的环境中，用同一 checkpoint/config/输入对比迁移前后参数键
和加载结果、固定区掩码、一次完整去噪、训练单步及双卡全 dummy 情况；随后对上面
P1 数值行为修复建立独立前后对照。当前不能据此声称论文结果已完整复现。
