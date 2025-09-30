# Table of Contents
- F:\SomeProjects\SpikeGNNNet2\.gitignore
- F:\SomeProjects\SpikeGNNNet2\generate_feature.py
- F:\SomeProjects\SpikeGNNNet2\LICENSE
- F:\SomeProjects\SpikeGNNNet2\main.py
- F:\SomeProjects\SpikeGNNNet2\main_optuna.py
- F:\SomeProjects\SpikeGNNNet2\main_static.py
- F:\SomeProjects\SpikeGNNNet2\project.md
- F:\SomeProjects\SpikeGNNNet2\readme.md
- F:\SomeProjects\SpikeGNNNet2\setup.py
- F:\SomeProjects\SpikeGNNNet2\spikenet\dataset.py
- F:\SomeProjects\SpikeGNNNet2\spikenet\deepwalk.py
- F:\SomeProjects\SpikeGNNNet2\spikenet\gates.py
- F:\SomeProjects\SpikeGNNNet2\spikenet\layers.py
- F:\SomeProjects\SpikeGNNNet2\spikenet\neuron.py
- F:\SomeProjects\SpikeGNNNet2\spikenet\sample_neighber.cpp
- F:\SomeProjects\SpikeGNNNet2\spikenet\temporal.py
- F:\SomeProjects\SpikeGNNNet2\spikenet\utils.py

## File: F:\SomeProjects\SpikeGNNNet2\.gitignore

- Extension: 
- Language: unknown
- Size: 1281 bytes
- Created: 2025-09-19 01:31:58
- Modified: 2023-09-27 17:42:24

### Code

```unknown
  1 | # Custom
  2 | *.idea
  3 | *.pdf
  4 | *.txt
  5 | *.npy
  6 | !requirements.txt
  7 | data/
  8 | # Byte-compiled / optimized / DLL files
  9 | __pycache__/
 10 | *.py[cod]
 11 | *$py.class
 12 | 
 13 | # C extensions
 14 | *.so
 15 | 
 16 | # Distribution / packaging
 17 | .Python
 18 | env/
 19 | build/
 20 | develop-eggs/
 21 | dist/
 22 | downloads/
 23 | eggs/
 24 | .eggs/
 25 | lib/
 26 | lib64/
 27 | parts/
 28 | sdist/
 29 | var/
 30 | *.egg-info/
 31 | .installed.cfg
 32 | *.egg
 33 | 
 34 | # PyInstaller
 35 | #  Usually these files are written by a python script from a template
 36 | #  before PyInstaller builds the exe, so as to inject date/other infos into it.
 37 | *.manifest
 38 | *.spec
 39 | 
 40 | # Installer logs
 41 | pip-log.txt
 42 | pip-delete-this-directory.txt
 43 | 
 44 | # Unit test / coverage reports
 45 | htmlcov/
 46 | .tox/
 47 | .coverage
 48 | .coverage.*
 49 | .cache
 50 | nosetests.xml
 51 | coverage.xml
 52 | *,cover
 53 | .hypothesis/
 54 | 
 55 | # Translations
 56 | *.mo
 57 | *.pot
 58 | 
 59 | # Django stuff:
 60 | *.log
 61 | local_settings.py
 62 | 
 63 | # Flask stuff:
 64 | instance/
 65 | .webassets-cache
 66 | 
 67 | # Scrapy stuff:
 68 | .scrapy
 69 | 
 70 | # Sphinx documentation
 71 | docs/build/
 72 | 
 73 | # PyBuilder
 74 | target/
 75 | 
 76 | # IPython Notebook
 77 | .ipynb_checkpoints
 78 | 
 79 | # pyenv
 80 | .python-version
 81 | 
 82 | # celery beat schedule file
 83 | celerybeat-schedule
 84 | 
 85 | # dotenv
 86 | .env
 87 | 
 88 | # virtualenv
 89 | venv/
 90 | ENV/
 91 | 
 92 | # Spyder project settings
 93 | .spyderproject
 94 | 
 95 | # Rope project settings
 96 | .ropeproject
 97 | 
 98 | *.pickle
 99 | .vscode
100 | 
101 | # checkpoint
102 | *.h5
103 | *.pkl
104 | *.pth
105 | 
106 | # Mac files
107 | .DS_Store
```

## File: F:\SomeProjects\SpikeGNNNet2\generate_feature.py

- Extension: .py
- Language: python
- Size: 1176 bytes
- Created: 2025-09-19 01:31:58
- Modified: 2023-09-27 17:42:24

### Code

```python
 1 | import argparse
 2 | 
 3 | import numpy as np
 4 | from tqdm import tqdm
 5 | 
 6 | from spikenet import dataset
 7 | from spikenet.deepwalk import DeepWalk
 8 | 
 9 | parser = argparse.ArgumentParser()
10 | parser.add_argument("--dataset", nargs="?", default="DBLP",
11 |                     help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
12 | parser.add_argument('--normalize', action='store_true',
13 |                     help='Whether to normalize output embedding. (default: False)')
14 | 
15 | 
16 | args = parser.parse_args()
17 | if args.dataset.lower() == "dblp":
18 |     data = dataset.DBLP()
19 | elif args.dataset.lower() == "tmall":
20 |     data = dataset.Tmall()
21 | elif args.dataset.lower() == "patent":
22 |     data = dataset.Patent()
23 | else:
24 |     raise ValueError(
25 |         f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")
26 | 
27 | 
28 | model = DeepWalk(80, 10, 128, window_size=10, negative=1, workers=16)
29 | xs = []
30 | for g in tqdm(data.adj):
31 |     model.fit(g)
32 |     x = model.get_embedding(normalize=args.normalize)
33 |     xs.append(x)
34 | 
35 | 
36 | file_path = f'{data.root}/{data.name}/{data.name}.npy'
37 | np.save(file_path, np.stack(xs, axis=0)) # [T, N, F]
38 | print(f"Generated node feautures saved at {file_path}")
```

## File: F:\SomeProjects\SpikeGNNNet2\LICENSE

- Extension: 
- Language: unknown
- Size: 1112 bytes
- Created: 2025-09-19 01:31:58
- Modified: 2023-09-27 17:42:24

### Code

```unknown
 1 | MIT License
 2 | 
 3 | Copyright (c) 2022 Jintang Li, Sun Yat-sen University
 4 | 
 5 | Permission is hereby granted, free of charge, to any person obtaining a copy
 6 | of this software and associated documentation files (the "Software"), to deal
 7 | in the Software without restriction, including without limitation the rights
 8 | to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 9 | copies of the Software, and to permit persons to whom the Software is
10 | furnished to do so, subject to the following conditions:
11 | 
12 | The above copyright notice and this permission notice shall be included in all
13 | copies or substantial portions of the Software.
14 | 
15 | THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
16 | IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
17 | FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
18 | AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
19 | LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
20 | OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
21 | SOFTWARE.
```

## File: F:\SomeProjects\SpikeGNNNet2\main.py

- Extension: .py
- Language: python
- Size: 14400 bytes
- Created: 2025-09-19 01:31:58
- Modified: 2025-09-19 04:24:11

### Code

```python
  1 | # File: main.py
  2 | import argparse
  3 | import time
  4 | 
  5 | import torch
  6 | import torch.nn as nn
  7 | from sklearn import metrics
  8 | from torch.utils.data import DataLoader
  9 | from tqdm import tqdm
 10 | 
 11 | from spikenet import dataset, neuron
 12 | from spikenet.layers import SAGEAggregator
 13 | from spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
 14 |                             set_seed, tab_printer)
 15 | from spikenet.temporal import GroupSparseDelay1D, TemporalSeparableReadout
 16 | from spikenet.gates import L2SGate
 17 | 
 18 | 
 19 | class SpikeNet(nn.Module):
 20 |     def __init__(self, in_features, out_features, hids=[32], alpha=1.0, p=0.5,
 21 |                  dropout=0.7, bias=True, aggr='mean', sampler='sage',
 22 |                  surrogate='triangle', sizes=[5, 2], concat=False, act='LIF',
 23 |                  use_gs_delay=True, delay_groups=8, delay_set=(1, 3, 5),
 24 |                  use_learnable_p=True, use_tsr=True):
 25 | 
 26 |         super().__init__()
 27 | 
 28 |         tau = 1.0
 29 |         if sampler == 'rw':
 30 |             self.sampler = [RandomWalkSampler(
 31 |                 add_selfloops(adj_matrix)) for adj_matrix in data.adj]
 32 |             self.sampler_t = [RandomWalkSampler(add_selfloops(
 33 |                 adj_matrix)) for adj_matrix in data.adj_evolve]
 34 |         elif sampler == 'sage':
 35 |             self.sampler = [Sampler(add_selfloops(adj_matrix))
 36 |                             for adj_matrix in data.adj]
 37 |             self.sampler_t = [Sampler(add_selfloops(adj_matrix))
 38 |                               for adj_matrix in data.adj_evolve]
 39 |         else:
 40 |             raise ValueError(sampler)
 41 | 
 42 |         aggregators, snn = nn.ModuleList(), nn.ModuleList()
 43 | 
 44 |         in_dim = in_features
 45 |         for hid in hids:
 46 |             aggregators.append(SAGEAggregator(in_dim, hid,
 47 |                                               concat=concat, bias=bias,
 48 |                                               aggr=aggr))
 49 | 
 50 |             if act == "IF":
 51 |                 snn.append(neuron.IF(alpha=alpha, surrogate=surrogate))
 52 |             elif act == 'LIF':
 53 |                 snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate))
 54 |             elif act == 'PLIF':
 55 |                 snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate))
 56 |             else:
 57 |                 raise ValueError(act)
 58 | 
 59 |             in_dim = hid * 2 if concat else hid
 60 | 
 61 |         self.aggregators = aggregators
 62 |         self.dropout = nn.Dropout(dropout)
 63 |         self.snn = snn
 64 |         self.sizes = sizes
 65 |         self.base_p = p
 66 |         self.use_gs_delay = use_gs_delay
 67 |         self.use_tsr = use_tsr
 68 |         self.use_learnable_p = use_learnable_p
 69 | 
 70 |         # 时间统计（不参与梯度）：新增边占比、平均度
 71 |         with torch.no_grad():
 72 |             stats = []
 73 |             for t in range(len(data)):
 74 |                 # edges: 累计；edges_evolve: 当期新增
 75 |                 e_now = data.edges_evolve[t].shape[1]
 76 |                 e_cum = data.edges[t].size(1)
 77 |                 r_new = float(e_now) / max(1.0, float(e_cum))
 78 |                 deg_mean = 2.0 * float(e_cum) / float(data.num_nodes)
 79 |                 stats.append([r_new, deg_mean])
 80 |             self.time_stats = torch.tensor(stats, dtype=torch.float32)
 81 | 
 82 |         # 可学习 p 门（每层一个标量，输入 F=2）
 83 |         if self.use_learnable_p:
 84 |             self.gate = L2SGate(num_layers=len(self.sizes), in_features=2, base_p=self.base_p)
 85 | 
 86 |         last_dim = in_dim  # 经过所有聚合后 root 节点的维度
 87 |         T = len(data)      # 时间步
 88 |         if self.use_gs_delay:
 89 |             self.delay = GroupSparseDelay1D(D=last_dim, T=T, groups=delay_groups, delays=delay_set)
 90 | 
 91 |         if self.use_tsr:
 92 |             self.readout = TemporalSeparableReadout(D=last_dim, C=out_features, k=5)
 93 |         else:
 94 |             self.pooling = nn.Linear(T * last_dim, out_features)
 95 | 
 96 |         # 供正则使用的缓存
 97 |         self._last_spikes_for_loss = None
 98 | 
 99 |     def _layer_forward(self, h_list, sizes):
100 |         """单时间步上的 SAGE+SNN 堆叠（沿用原有实现）。"""
101 |         for i, aggregator in enumerate(self.aggregators):
102 |             self_x = h_list[:-1]
103 |             neigh_x = []
104 |             for j, n_x in enumerate(h_list[1:]):
105 |                 neigh_x.append(n_x.view(-1, sizes[j], h_list[0].size(-1)))
106 |             out = self.snn[i](aggregator(self_x, neigh_x))
107 |             if i != len(sizes) - 1:
108 |                 out = self.dropout(out)
109 |                 # 更新到下一层的多跳特征切片
110 |                 # num_nodes 在外层维护，这里由调用方传入切分信息
111 |             h_list = None  # 防止误用
112 |         return out
113 | 
114 |     def encode(self, nodes: torch.Tensor):
115 |         spikes_per_t = []
116 |         sizes = self.sizes
117 |         for t in range(len(data)):
118 |             snapshot = data[t]
119 |             sampler = self.sampler[t]
120 |             sampler_t = self.sampler_t[t]
121 | 
122 |             x = snapshot.x
123 |             h = [x[nodes].to(device)]
124 |             num_nodes = [nodes.size(0)]
125 |             nbr = nodes
126 | 
127 |             # 每层的 p_{l,t}
128 |             if self.use_learnable_p:
129 |                 p_vec = self.gate(self.time_stats[t]).cpu().tolist()  # [L]
130 |             else:
131 |                 p_vec = [self.base_p for _ in sizes]
132 | 
133 |             # 逐层扩展采样邻居
134 |             for li, size in enumerate(sizes):
135 |                 p_now = float(p_vec[li])
136 |                 size_1 = max(int(size * p_now), 1)
137 |                 size_2 = size - size_1
138 |                 if size_2 > 0:
139 |                     nbr_1 = sampler(nbr, size_1).view(nbr.size(0), size_1)
140 |                     nbr_2 = sampler_t(nbr, size_2).view(nbr.size(0), size_2)
141 |                     nbr = torch.cat([nbr_1, nbr_2], dim=1).flatten()
142 |                 else:
143 |                     nbr = sampler(nbr, size_1).view(-1)
144 |                 num_nodes.append(nbr.size(0))
145 |                 h.append(x[nbr].to(device))
146 | 
147 |             # SAGE + SNN 堆叠
148 |             for i, aggregator in enumerate(self.aggregators):
149 |                 self_x = h[:-1]
150 |                 neigh_x = []
151 |                 for j, n_x in enumerate(h[1:]):
152 |                     neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))
153 |                 out = self.snn[i](aggregator(self_x, neigh_x))
154 |                 if i != len(sizes) - 1:
155 |                     out = self.dropout(out)
156 |                     h = torch.split(out, num_nodes[:-(i + 1)])
157 | 
158 |             spikes_per_t.append(out)  # [N, D]
159 | 
160 |         # [N, T, D]
161 |         spikes = torch.stack(spikes_per_t, dim=1)
162 |         if self.use_gs_delay:
163 |             spikes = self.delay(spikes)
164 |         self._last_spikes_for_loss = spikes
165 |         neuron.reset_net(self)
166 |         return spikes
167 | 
168 |     def forward(self, nodes):
169 |         spikes = self.encode(nodes)  # [B, T, D]
170 |         if self.use_tsr:
171 |             return self.readout(spikes)
172 |         else:
173 |             return self.pooling(spikes.flatten(1))
174 | 
175 | 
176 | parser = argparse.ArgumentParser()
177 | parser.add_argument("--dataset", nargs="?", default="DBLP",
178 |                     help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
179 | parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2], help='Neighborhood sampling size for each layer. (default: [5, 2])')
180 | parser.add_argument('--hids', type=int, nargs='+',
181 |                     default=[128, 10], help='Hidden units for each layer. (default: [128, 10])')
182 | parser.add_argument("--aggr", nargs="?", default="mean",
183 |                     help="Aggregate function ('mean', 'sum'). (default: 'mean')")
184 | parser.add_argument("--sampler", nargs="?", default="sage",
185 |                     help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
186 | parser.add_argument("--surrogate", nargs="?", default="sigmoid",
187 |                     help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
188 | parser.add_argument("--neuron", nargs="?", default="LIF",
189 |                     help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
190 | parser.add_argument('--batch_size', type=int, default=1024,
191 |                     help='Batch size for training. (default: 1024)')
192 | parser.add_argument('--lr', type=float, default=5e-3,
193 |                     help='Learning rate for training. (default: 5e-3)')
194 | parser.add_argument('--train_size', type=float, default=0.4,
195 |                     help='Ratio of nodes for training. (default: 0.4)')
196 | parser.add_argument('--alpha', type=float, default=1.0,
197 |                     help='Smooth factor for surrogate learning. (default: 1.0)')
198 | parser.add_argument('--p', type=float, default=0.5,
199 |                     help='Percentage of sampled neighborhoods for g_t. (default: 0.5)')
200 | parser.add_argument('--dropout', type=float, default=0.7,
201 |                     help='Dropout probability. (default: 0.7)')
202 | parser.add_argument('--epochs', type=int, default=100,
203 |                     help='Number of training epochs. (default: 100)')
204 | parser.add_argument('--concat', action='store_true',
205 |                     help='Whether to concat node representation and neighborhood representations. (default: False)')
206 | parser.add_argument('--seed', type=int, default=2022,
207 |                     help='Random seed for model. (default: 2022)')
208 | # 新增功能开关
209 | parser.add_argument('--use_gs_delay', type=int, default=1, help='Use Group-Sparse Delay kernel (1|0).')
210 | parser.add_argument('--delay_groups', type=int, default=8, help='Delay groups for GS-Delay.')
211 | parser.add_argument('--delays', type=int, nargs='+', default=[1,3,5], help='Sparse delay set Δ.')
212 | parser.add_argument('--use_tsr', type=int, default=1, help='Use Temporal Separable Readout (1|0).')
213 | parser.add_argument('--use_learnable_p', type=int, default=1, help='Use learnable sampling gate (1|0).')
214 | # 正则与课程
215 | parser.add_argument('--spike_reg', type=float, default=0.0, help='Avg firing-rate regularization λ.')
216 | parser.add_argument('--temp_reg', type=float, default=0.0, help='Temporal consistency regularization λ.')
217 | parser.add_argument('--alpha_warmup', type=float, default=0.3, help='Warm-up ratio of epochs for surrogate alpha (0~1).')
218 | 
219 | try:
220 |     args = parser.parse_args()
221 |     args.test_size = 1 - args.train_size
222 |     args.train_size = args.train_size - 0.05
223 |     args.val_size = 0.05
224 |     args.split_seed = 42
225 |     tab_printer(args)
226 | except:
227 |     parser.print_help()
228 |     exit(0)
229 | 
230 | assert len(args.hids) == len(args.sizes), "must be equal!"
231 | 
232 | if args.dataset.lower() == "dblp":
233 |     data = dataset.DBLP(root='/data4/zhengzhuoyu/data')
234 | elif args.dataset.lower() == "tmall":
235 |     data = dataset.Tmall(root='/data4/zhengzhuoyu/data')
236 | elif args.dataset.lower() == "patent":
237 |     data = dataset.Patent(root='/data4/zhengzhuoyu/data')
238 | else:
239 |     raise ValueError(
240 |         f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")
241 | 
242 | # train:val:test
243 | data.split_nodes(train_size=args.train_size, val_size=args.val_size,
244 |                  test_size=args.test_size, random_state=args.split_seed)
245 | 
246 | set_seed(args.seed)
247 | 
248 | device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
249 | 
250 | y = data.y.to(device)
251 | 
252 | train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=True)
253 | val_loader = DataLoader(data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist(),
254 |                         pin_memory=False, batch_size=200000, shuffle=False)
255 | test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=200000, shuffle=False)
256 | 
257 | model = SpikeNet(data.num_features, data.num_classes, alpha=args.alpha,
258 |                  dropout=args.dropout, sampler=args.sampler, p=args.p,
259 |                  aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
260 |                  hids=args.hids, act=args.neuron, bias=True,
261 |                  use_gs_delay=bool(args.use_gs_delay), delay_groups=args.delay_groups,
262 |                  delay_set=tuple(args.delays), use_learnable_p=bool(args.use_learnable_p),
263 |                  use_tsr=bool(args.use_tsr)).to(device)
264 | 
265 | optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
266 | loss_fn = nn.CrossEntropyLoss()
267 | 
268 | 
269 | def set_alpha_with_warmup(net: nn.Module, base_alpha: float, epoch: int, total_epochs: int, warmup_ratio: float):
270 |     if warmup_ratio <= 0:
271 |         cur_alpha = base_alpha
272 |     else:
273 |         warm_epochs = max(1, int(total_epochs * warmup_ratio))
274 |         factor = min(1.0, float(epoch) / float(warm_epochs))
275 |         cur_alpha = max(1e-3, base_alpha * factor)
276 |     # 将 alpha 写入所有神经元模块
277 |     for m in net.modules():
278 |         if hasattr(m, 'alpha') and isinstance(m.alpha, torch.Tensor):
279 |             m.alpha.data.fill_(cur_alpha)
280 | 
281 | 
282 | def train():
283 |     model.train()
284 |     for nodes in tqdm(train_loader, desc='Training'):
285 |         optimizer.zero_grad()
286 |         logits = model(nodes)
287 |         ce = loss_fn(logits, y[nodes])
288 |         loss = ce
289 |         # 正则项
290 |         spikes = model._last_spikes_for_loss
291 |         if spikes is not None:
292 |             if args.spike_reg > 0:
293 |                 reg_spike = spikes.mean()
294 |                 loss = loss + args.spike_reg * reg_spike
295 |             if args.temp_reg > 0 and spikes.size(1) > 1:
296 |                 reg_temp = (spikes[:, 1:] - spikes[:, :-1]).pow(2).mean()
297 |                 loss = loss + args.temp_reg * reg_temp
298 |         loss.backward()
299 |         optimizer.step()
300 | 
301 | 
302 | @torch.no_grad()
303 | def test(loader):
304 |     model.eval()
305 |     logits = []
306 |     labels = []
307 |     for nodes in loader:
308 |         logits.append(model(nodes))
309 |         labels.append(y[nodes])
310 |     logits = torch.cat(logits, dim=0).cpu()
311 |     labels = torch.cat(labels, dim=0).cpu()
312 |     preds = logits.argmax(1)
313 |     metric_macro = metrics.f1_score(labels, preds, average='macro')
314 |     metric_micro = metrics.f1_score(labels, preds, average='micro')
315 |     return metric_macro, metric_micro
316 | 
317 | 
318 | best_val_metric = test_metric = 0
319 | start = time.time()
320 | for epoch in range(1, args.epochs + 1):
321 |     # surrogate alpha 课程
322 |     set_alpha_with_warmup(model, args.alpha, epoch, args.epochs, args.alpha_warmup)
323 |     train()
324 |     val_metric, test_metric = test(val_loader), test(test_loader)
325 |     if val_metric[1] > best_val_metric:
326 |         best_val_metric = val_metric[1]
327 |         best_test_metric = test_metric
328 |     end = time.time()
329 |     print(
330 |         f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, '
331 |         f'Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s')
332 | 
333 | # # 如需保存二进制脉冲表示（spikes）
334 | # # emb = model.encode(torch.arange(data.num_nodes)).cpu()
335 | # # torch.save(emb, 'emb.pth')
```

## File: F:\SomeProjects\SpikeGNNNet2\main_optuna.py

- Extension: .py
- Language: python
- Size: 18274 bytes
- Created: 2025-09-19 05:35:48
- Modified: 2025-09-22 00:12:10

### Code

```python
  1 | # File: main_optuna.py
  2 | import argparse
  3 | import time
  4 | import warnings
  5 | from typing import Tuple
  6 | 
  7 | import optuna
  8 | from optuna.trial import TrialState
  9 | 
 10 | import torch
 11 | import torch.nn as nn
 12 | from torch.utils.data import DataLoader
 13 | from sklearn import metrics
 14 | 
 15 | # === 你项目里的模块 ===
 16 | from spikenet import dataset, neuron
 17 | from spikenet.layers import SAGEAggregator
 18 | from spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
 19 |                             set_seed)
 20 | from spikenet.temporal import GroupSparseDelay1D, TemporalSeparableReadout
 21 | from spikenet.gates import L2SGate
 22 | 
 23 | warnings.filterwarnings("ignore", category=UserWarning)
 24 | 
 25 | # --------- 全局（为兼容你模型里对全局变量 data/device 的引用） ---------
 26 | data = None          # will be set in build_data(...)
 27 | device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 28 | y = None
 29 | 
 30 | 
 31 | # ========================= 你的模型（与 main.py 基本一致） =========================
 32 | class SpikeNet(nn.Module):
 33 |     def __init__(self, in_features, out_features, hids=[32], alpha=1.0, p=0.5,
 34 |                  dropout=0.7, bias=True, aggr='mean', sampler='sage',
 35 |                  surrogate='triangle', sizes=[5, 2], concat=False, act='LIF',
 36 |                  use_gs_delay=True, delay_groups=8, delay_set=(1, 3, 5),
 37 |                  use_learnable_p=True, use_tsr=True):
 38 | 
 39 |         super().__init__()
 40 | 
 41 |         tau = 1.0
 42 |         if sampler == 'rw':
 43 |             self.sampler = [RandomWalkSampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj]
 44 |             self.sampler_t = [RandomWalkSampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj_evolve]
 45 |         elif sampler == 'sage':
 46 |             self.sampler = [Sampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj]
 47 |             self.sampler_t = [Sampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj_evolve]
 48 |         else:
 49 |             raise ValueError(sampler)
 50 | 
 51 |         aggregators, snn = nn.ModuleList(), nn.ModuleList()
 52 | 
 53 |         in_dim = in_features
 54 |         for hid in hids:
 55 |             aggregators.append(SAGEAggregator(in_dim, hid, concat=concat, bias=bias, aggr=aggr))
 56 | 
 57 |             if act == "IF":
 58 |                 snn.append(neuron.IF(alpha=alpha, surrogate=surrogate))
 59 |             elif act == 'LIF':
 60 |                 snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate))
 61 |             elif act == 'PLIF':
 62 |                 snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate))
 63 |             else:
 64 |                 raise ValueError(act)
 65 | 
 66 |             in_dim = hid * 2 if concat else hid
 67 | 
 68 |         self.aggregators = aggregators
 69 |         self.dropout = nn.Dropout(dropout)
 70 |         self.snn = snn
 71 |         self.sizes = sizes
 72 |         self.base_p = p
 73 |         self.use_gs_delay = use_gs_delay
 74 |         self.use_tsr = use_tsr
 75 |         self.use_learnable_p = use_learnable_p
 76 | 
 77 |         # 时间统计（不参与梯度）：新增边占比、平均度
 78 |         with torch.no_grad():
 79 |             stats = []
 80 |             for t in range(len(data)):
 81 |                 e_now = data.edges_evolve[t].shape[1]
 82 |                 e_cum = data.edges[t].size(1)
 83 |                 r_new = float(e_now) / max(1.0, float(e_cum))
 84 |                 deg_mean = 2.0 * float(e_cum) / float(data.num_nodes)
 85 |                 stats.append([r_new, deg_mean])
 86 |             self.time_stats = torch.tensor(stats, dtype=torch.float32)
 87 | 
 88 |         # 可学习 p 门
 89 |         if self.use_learnable_p:
 90 |             self.gate = L2SGate(num_layers=len(self.sizes), in_features=2, base_p=self.base_p)
 91 | 
 92 |         last_dim = in_dim
 93 |         T = len(data)
 94 |         if self.use_gs_delay:
 95 |             self.delay = GroupSparseDelay1D(D=last_dim, T=T, groups=delay_groups, delays=delay_set)
 96 | 
 97 |         if self.use_tsr:
 98 |             self.readout = TemporalSeparableReadout(D=last_dim, C=out_features, k=5)
 99 |         else:
100 |             self.pooling = nn.Linear(T * last_dim, out_features)
101 | 
102 |         self._last_spikes_for_loss = None
103 | 
104 |     def encode(self, nodes: torch.Tensor):
105 |         spikes_per_t = []
106 |         sizes = self.sizes
107 |         for t in range(len(data)):
108 |             snapshot = data[t]
109 |             sampler = self.sampler[t]
110 |             sampler_t = self.sampler_t[t]
111 | 
112 |             x = snapshot.x
113 |             h = [x[nodes].to(device)]
114 |             num_nodes = [nodes.size(0)]
115 |             nbr = nodes
116 | 
117 |             # 每层的 p_{l,t}
118 |             if self.use_learnable_p:
119 |                 p_vec = self.gate(self.time_stats[t]).cpu().tolist()  # [L]
120 |             else:
121 |                 p_vec = [self.base_p for _ in sizes]
122 | 
123 |             # 逐层扩展采样邻居
124 |             for li, size in enumerate(sizes):
125 |                 p_now = float(p_vec[li])
126 |                 size_1 = max(int(size * p_now), 1)
127 |                 size_2 = size - size_1
128 |                 if size_2 > 0:
129 |                     nbr_1 = sampler(nbr, size_1).view(nbr.size(0), size_1)
130 |                     nbr_2 = sampler_t(nbr, size_2).view(nbr.size(0), size_2)
131 |                     nbr = torch.cat([nbr_1, nbr_2], dim=1).flatten()
132 |                 else:
133 |                     nbr = sampler(nbr, size_1).view(-1)
134 |                 num_nodes.append(nbr.size(0))
135 |                 h.append(x[nbr].to(device))
136 | 
137 |             # SAGE + SNN 堆叠
138 |             for i, aggregator in enumerate(self.aggregators):
139 |                 self_x = h[:-1]
140 |                 neigh_x = []
141 |                 for j, n_x in enumerate(h[1:]):
142 |                     neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))
143 |                 out = self.snn[i](aggregator(self_x, neigh_x))
144 |                 if i != len(sizes) - 1:
145 |                     out = self.dropout(out)
146 |                     h = torch.split(out, num_nodes[:-(i + 1)])
147 |             spikes_per_t.append(out)  # [N, D]
148 | 
149 |         spikes = torch.stack(spikes_per_t, dim=1)  # [N, T, D]
150 |         if self.use_gs_delay:
151 |             spikes = self.delay(spikes)
152 |         self._last_spikes_for_loss = spikes
153 |         neuron.reset_net(self)
154 |         return spikes
155 | 
156 |     def forward(self, nodes):
157 |         spikes = self.encode(nodes)  # [B, T, D]
158 |         if self.use_tsr:
159 |             return self.readout(spikes)
160 |         else:
161 |             return self.pooling(spikes.flatten(1))
162 | 
163 | 
164 | # ========================= 训练/评估工具 =========================
165 | def set_alpha_with_warmup(net: nn.Module, base_alpha: float, epoch: int, total_epochs: int, warmup_ratio: float):
166 |     if warmup_ratio <= 0:
167 |         cur_alpha = base_alpha
168 |     else:
169 |         warm_epochs = max(1, int(total_epochs * warmup_ratio))
170 |         factor = min(1.0, float(epoch) / float(warm_epochs))
171 |         cur_alpha = max(1e-3, base_alpha * factor)
172 |     for m in net.modules():
173 |         if hasattr(m, 'alpha') and isinstance(m.alpha, torch.Tensor):
174 |             m.alpha.data.fill_(cur_alpha)
175 | 
176 | 
177 | def train_one_epoch(model, optimizer, loss_fn, train_loader, spike_reg=0.0, temp_reg=0.0):
178 |     model.train()
179 |     for nodes in train_loader:
180 |         optimizer.zero_grad()
181 |         logits = model(nodes)
182 |         ce = loss_fn(logits, y[nodes])
183 |         loss = ce
184 |         # 正则
185 |         spikes = model._last_spikes_for_loss
186 |         if spikes is not None:
187 |             if spike_reg > 0:
188 |                 loss = loss + spike_reg * spikes.mean()
189 |             if temp_reg > 0 and spikes.size(1) > 1:
190 |                 loss = loss + temp_reg * (spikes[:, 1:] - spikes[:, :-1]).pow(2).mean()
191 |         loss.backward()
192 |         optimizer.step()
193 | 
194 | 
195 | @torch.no_grad()
196 | def evaluate(model, loader) -> Tuple[float, float]:
197 |     model.eval()
198 |     logits, labels = [], []
199 |     for nodes in loader:
200 |         logits.append(model(nodes))
201 |         labels.append(y[nodes])
202 |     logits = torch.cat(logits, dim=0).cpu()
203 |     labels = torch.cat(labels, dim=0).cpu()
204 |     preds = logits.argmax(1)
205 |     macro = metrics.f1_score(labels, preds, average='macro')
206 |     micro = metrics.f1_score(labels, preds, average='micro')
207 |     return macro, micro
208 | 
209 | 
210 | # ========================= 数据 & Loader =========================
211 | def build_data(dataset_name: str, root: str, train_size: float, val_size: float, split_seed: int):
212 |     global data, y
213 |     name = dataset_name.lower()
214 |     if name == "dblp":
215 |         data = dataset.DBLP(root=root)
216 |     elif name == "tmall":
217 |         data = dataset.Tmall(root=root)
218 |     elif name == "patent":
219 |         data = dataset.Patent(root=root)
220 |     else:
221 |         raise ValueError(f"{dataset_name} is invalid. Only datasets (dblp, tmall, patent) are available.")
222 | 
223 |     # train : val : test
224 |     data.split_nodes(train_size=train_size, val_size=val_size,
225 |                      test_size=1 - train_size - val_size, random_state=split_seed)
226 |     y = data.y.to(device)
227 | 
228 | 
229 | def make_loaders(batch_size: int):
230 |     train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=batch_size, shuffle=True)
231 |     val_nodes = data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist()
232 |     val_loader = DataLoader(val_nodes, pin_memory=False, batch_size=200000, shuffle=False)
233 |     test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=200000, shuffle=False)
234 |     return train_loader, val_loader, test_loader
235 | 
236 | 
237 | # ========================= Optuna 目标函数 =========================
238 | def build_model_from_trial(trial, in_features, out_features):
239 |     # 层数
240 |     n_layers = trial.suggest_int("n_layers", 2, 3)
241 | 
242 |     # 每层采样邻居数（保持参数名稳定）
243 |     sizes = []
244 |     for i in range(n_layers):
245 |         if i == 0:
246 |             sizes.append(trial.suggest_int(f"size_l{i}", 4, 15))
247 |         else:
248 |             sizes.append(trial.suggest_int(f"size_l{i}", 2, 10))
249 | 
250 |     # 固定隐藏维度
251 |     hids = [128] * n_layers
252 | 
253 |     # 其它超参（choices 全是标量）
254 |     dropout = trial.suggest_float("dropout", 0.3, 0.8)
255 |     lr = trial.suggest_float("lr", 1e-5, 5e-2, log=True)
256 |     alpha = trial.suggest_float("alpha", 0.5, 2.0)
257 |     alpha_warmup = trial.suggest_float("alpha_warmup", 0.2, 0.5)
258 |     p = trial.suggest_float("p_base", 0.6, 0.95)
259 | 
260 |     sampler = trial.suggest_categorical("sampler", ("sage", "rw"))
261 |     aggr = trial.suggest_categorical("aggr", ("mean", "sum"))
262 |     surrogate = trial.suggest_categorical("surrogate", ("sigmoid", "triangle", "arctan", "mg", "super"))
263 |     neuron_type = trial.suggest_categorical("neuron", ("IF", "LIF", "PLIF"))
264 |     concat = trial.suggest_categorical("concat", (False, True))
265 | 
266 |     use_learnable_p = trial.suggest_categorical("use_learnable_p", (True, False))
267 |     use_gs_delay   = trial.suggest_categorical("use_gs_delay", (True, False))
268 |     use_tsr        = trial.suggest_categorical("use_tsr", (True, False))
269 | 
270 |     # === 关键改动 1：始终 suggest（避免动态空间） ===
271 |     delay_groups_suggested = trial.suggest_int("delay_groups", 4, 16)
272 | 
273 |     # === 关键改动 2：choices 只用“标量 id”，再映射到 tuple ===
274 |     #   0 -> (0,1,2), 1 -> (0,1,3,5), 2 -> (1,3,5)
275 |     delay_set_id = trial.suggest_categorical("delay_set_id_v3", (0, 1, 2))
276 |     DELAY_SETS = {
277 |         0: (0, 1, 2),
278 |         1: (0, 1, 3, 5),
279 |         2: (1, 3, 5),
280 |     }
281 |     delay_set_suggested = DELAY_SETS[int(delay_set_id)]
282 | 
283 |     # 未启用时使用固定有效值，但搜索空间保持不变
284 |     delay_groups_eff = delay_groups_suggested if use_gs_delay else 8
285 |     delay_set_eff    = delay_set_suggested   if use_gs_delay else (1, 3, 5)
286 | 
287 |     # 正则
288 |     spike_reg = trial.suggest_categorical("spike_reg", (0.0, 1e-5, 1e-4, 5e-4, 1e-3))
289 |     temp_reg  = trial.suggest_categorical("temp_reg",  (0.0, 1e-6, 5e-6, 1e-5, 5e-5))
290 | 
291 |     batch_size = 1024
292 | 
293 |     model = SpikeNet(
294 |         in_features, out_features,
295 |         hids=hids, alpha=alpha, p=p,
296 |         dropout=dropout, bias=True, aggr=aggr, sampler=sampler,
297 |         surrogate=surrogate, sizes=sizes, concat=concat, act=neuron_type,
298 |         use_gs_delay=use_gs_delay,
299 |         delay_groups=delay_groups_eff,
300 |         delay_set=delay_set_eff,
301 |         use_learnable_p=use_learnable_p, use_tsr=use_tsr
302 |     ).to(device)
303 | 
304 |     # 记录“有效值”到 dashboard
305 |     trial.set_user_attr("gs_delay_enabled", bool(use_gs_delay))
306 |     trial.set_user_attr("delay_groups_eff", int(delay_groups_eff))
307 |     trial.set_user_attr("delay_set_eff", tuple(delay_set_eff))
308 | 
309 |     cfg = dict(
310 |         lr=lr, alpha=alpha, alpha_warmup=alpha_warmup,
311 |         spike_reg=spike_reg, temp_reg=temp_reg, batch_size=batch_size
312 |     )
313 |     return model, cfg
314 | 
315 | 
316 | 
317 | def objective_factory(args):
318 |     # 数据只加载一次，所有 trial 共享同一拆分，保证公平
319 |     build_data(args.dataset, args.root, train_size=args.train_size, val_size=args.val_size, split_seed=args.split_seed)
320 | 
321 |     def objective(trial: optuna.trial.Trial):
322 |         set_seed(args.seed + trial.number)
323 | 
324 |         model, cfg = build_model_from_trial(trial, data.num_features, data.num_classes)
325 |         optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
326 |         loss_fn = nn.CrossEntropyLoss()
327 | 
328 |         # 局部 loader
329 |         train_loader, val_loader, test_loader = make_loaders(cfg["batch_size"])
330 | 
331 |         best_val_micro = -1.0
332 |         best_test_macro = 0.0
333 |         best_test_micro = 0.0
334 | 
335 |         t0 = time.time()
336 |         for epoch in range(1, args.epochs + 1):
337 |             set_alpha_with_warmup(model, cfg["alpha"], epoch, args.epochs, cfg["alpha_warmup"])
338 |             train_one_epoch(model, optimizer, loss_fn, train_loader,
339 |                             spike_reg=cfg["spike_reg"], temp_reg=cfg["temp_reg"])
340 | 
341 |             val_macro, val_micro = evaluate(model, val_loader)
342 |             test_macro, test_micro = evaluate(model, test_loader)
343 | 
344 |             # 向 Optuna 汇报中间结果，并用于裁剪
345 |             trial.report(val_micro, step=epoch)
346 | 
347 |             # 保存最好验证对应的测试指标
348 |             if val_micro > best_val_micro:
349 |                 best_val_micro = val_micro
350 |                 best_test_macro = test_macro
351 |                 best_test_micro = test_micro
352 | 
353 |             if trial.should_prune():
354 |                 # 把当前最优记录挂到 user_attrs，便于 dashboard 上查看
355 |                 trial.set_user_attr("best_test_macro_sofar", float(best_test_macro))
356 |                 trial.set_user_attr("best_test_micro_sofar", float(best_test_micro))
357 |                 raise optuna.exceptions.TrialPruned()
358 | 
359 |         # 结束后把关键统计写入 user_attrs
360 |         trial.set_user_attr("best_val_micro", float(best_val_micro))
361 |         trial.set_user_attr("best_test_macro", float(best_test_macro))
362 |         trial.set_user_attr("best_test_micro", float(best_test_micro))
363 |         trial.set_user_attr("epochs", int(args.epochs))
364 |         trial.set_user_attr("train_time_sec", float(time.time() - t0))
365 | 
366 |         # 目标：最大化验证 Micro-F1
367 |         return best_val_micro
368 | 
369 |     return objective
370 | 
371 | 
372 | # ========================= CLI & Study =========================
373 | def parse_args():
374 |     p = argparse.ArgumentParser("SpikeNet Optuna Search")
375 |     # 数据与拆分
376 |     p.add_argument("--dataset", default="DBLP", help="DBLP | Tmall | Patent")
377 |     p.add_argument("--root", default="/data4/zhengzhuoyu/data")
378 |     p.add_argument("--train_size", type=float, default=0.8)
379 |     p.add_argument("--val_size", type=float, default=0.05)
380 |     p.add_argument("--split_seed", type=int, default=42)
381 |     p.add_argument("--seed", type=int, default=2022)
382 | 
383 |     # 训练轮数
384 |     p.add_argument("--epochs", type=int, default=60)
385 | 
386 |     # Optuna
387 |     # 用新名字避免旧 Study（含 list choices）影响 Dashboard
388 |     p.add_argument("--study", default="SpikeNet-HPO-v3")
389 |     p.add_argument("--storage", default="sqlite:///s_optuna.db",
390 |                    help="sqlite:///file.db 或 mysql://user:pwd@host/db 等")
391 |     p.add_argument("--n-trials", type=int, default=100)
392 |     p.add_argument("--timeout", type=int, default=None)
393 | 
394 |     # 采样器/裁剪器
395 |     p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
396 |     p.add_argument("--pruner", choices=["median", "sha", "none"], default="median")
397 | 
398 |     return p.parse_args()
399 | 
400 | 
401 | def make_study(args):
402 |     if args.sampler == "tpe":
403 |         sampler = optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=10)
404 |     else:
405 |         sampler = optuna.samplers.RandomSampler()
406 | 
407 |     if args.pruner == "median":
408 |         pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(5, args.epochs // 5),
409 |                                              n_min_trials=5)
410 |     elif args.pruner == "sha":
411 |         pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=max(5, args.epochs // 6),
412 |                                                         reduction_factor=3)
413 |     else:
414 |         pruner = optuna.pruners.NopPruner()
415 | 
416 |     study = optuna.create_study(direction="maximize",
417 |                                 study_name=args.study,
418 |                                 storage=args.storage,
419 |                                 load_if_exists=True,
420 |                                 sampler=sampler,
421 |                                 pruner=pruner)
422 |     return study
423 | 
424 | 
425 | def main():
426 |     args = parse_args()
427 |     print(f"[Info] Using device: {device}")
428 |     objective = objective_factory(args)
429 |     study = make_study(args)
430 | 
431 |     study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
432 | 
433 |     pruned = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
434 |     complete = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
435 | 
436 |     print("Study statistics:")
437 |     print(f"  Finished trials: {len(study.trials)}")
438 |     print(f"  Pruned trials:   {len(pruned)}")
439 |     print(f"  Complete trials: {len(complete)}")
440 | 
441 |     print("Best trial:")
442 |     best = study.best_trial
443 |     print(f"  Value (Val Micro-F1): {best.value:.4f}")
444 |     print("  Params:")
445 |     for k, v in best.params.items():
446 |         print(f"    {k}: {v}")
447 |     # 同时把 Test Macro/Micro 打印出来（如果存在）
448 |     bm = best.user_attrs.get("best_test_macro")
449 |     bmi = best.user_attrs.get("best_test_micro")
450 |     if bm is not None and bmi is not None:
451 |         print(f"  Linked Test (Macro, Micro): ({bm:.4f}, {bmi:.4f})")
452 | 
453 | 
454 | if __name__ == "__main__":
455 |     main()
```

## File: F:\SomeProjects\SpikeGNNNet2\main_static.py

- Extension: .py
- Language: python
- Size: 11251 bytes
- Created: 2025-09-19 01:31:58
- Modified: 2025-09-19 04:12:06

### Code

```python
  1 | # File: main_static.py
  2 | import argparse
  3 | import os.path as osp
  4 | import time
  5 | 
  6 | import torch
  7 | import torch.nn as nn
  8 | from sklearn import metrics
  9 | from spikenet import dataset, neuron
 10 | from spikenet.layers import SAGEAggregator
 11 | from spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
 12 |                             set_seed, tab_printer)
 13 | from torch.utils.data import DataLoader
 14 | from torch_geometric.datasets import Flickr, Reddit
 15 | from torch_geometric.utils import to_scipy_sparse_matrix
 16 | from tqdm import tqdm
 17 | 
 18 | from spikenet.temporal import GroupSparseDelay1D, TemporalSeparableReadout
 19 | from spikenet.gates import L2SGate
 20 | 
 21 | 
 22 | class SpikeNet(nn.Module):
 23 |     def __init__(self, in_features, out_features, hids=[32], alpha=1.0, T=5,
 24 |                  dropout=0.7, bias=True, aggr='mean', sampler='sage',
 25 |                  surrogate='triangle', sizes=[5, 2], concat=False, act='LIF',
 26 |                  use_gs_delay=True, delay_groups=8, delay_set=(1,3,5), use_tsr=True):
 27 | 
 28 |         super().__init__()
 29 | 
 30 |         tau = 1.0
 31 |         if sampler == 'rw':
 32 |             self.sampler = RandomWalkSampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
 33 |         elif sampler == 'sage':
 34 |             self.sampler = Sampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
 35 |         else:
 36 |             raise ValueError(sampler)
 37 | 
 38 |         del data.edge_index
 39 | 
 40 |         aggregators, snn = nn.ModuleList(), nn.ModuleList()
 41 | 
 42 |         in_dim = in_features
 43 |         for hid in hids:
 44 |             aggregators.append(SAGEAggregator(in_dim, hid,
 45 |                                               concat=concat, bias=bias,
 46 |                                               aggr=aggr))
 47 |             if act == "IF":
 48 |                 snn.append(neuron.IF(alpha=alpha, surrogate=surrogate))
 49 |             elif act == 'LIF':
 50 |                 snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate))
 51 |             elif act == 'PLIF':
 52 |                 snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate))
 53 |             else:
 54 |                 raise ValueError(act)
 55 |             in_dim = hid * 2 if concat else hid
 56 | 
 57 |         self.aggregators = aggregators
 58 |         self.dropout = nn.Dropout(dropout)
 59 |         self.snn = snn
 60 |         self.sizes = sizes
 61 |         self.T = T
 62 |         self.use_gs_delay = use_gs_delay
 63 |         self.use_tsr = use_tsr
 64 | 
 65 |         last_dim = in_dim
 66 |         if self.use_gs_delay:
 67 |             self.delay = GroupSparseDelay1D(D=last_dim, T=T, groups=delay_groups, delays=delay_set)
 68 |         if self.use_tsr:
 69 |             self.readout = TemporalSeparableReadout(D=last_dim, C=out_features, k=5)
 70 |         else:
 71 |             self.pooling = nn.Linear(T * last_dim, out_features)
 72 | 
 73 |         # 退化版门控：无时间统计时，等价为每层一个可学习标量（通过 bias 表现）
 74 |         self.gate = L2SGate(num_layers=len(self.sizes), in_features=2, base_p=0.5)
 75 |         self._zero_stats = torch.zeros(2)
 76 | 
 77 |         self._last_spikes_for_loss = None
 78 | 
 79 |     def encode(self, nodes):
 80 |         spikes = []
 81 |         sizes = self.sizes
 82 |         x = data.x
 83 | 
 84 |         for time_step in range(self.T):
 85 |             h = [x[nodes].to(device)]
 86 |             num_nodes = [nodes.size(0)]
 87 |             nbr = nodes
 88 | 
 89 |             p_vec = self.gate(self._zero_stats).cpu().tolist()
 90 | 
 91 |             for li, size in enumerate(sizes):
 92 |                 p_now = float(p_vec[li])
 93 |                 size_1 = max(int(size * p_now), 1)
 94 |                 size_2 = size - size_1
 95 |                 if size_2 > 0:
 96 |                     nbr_1 = self.sampler(nbr, size_1)
 97 |                     nbr_2 = self.sampler(nbr, size_2)  # 静态图无演化，用同一采样器
 98 |                     nbr = torch.cat([nbr_1.view(nbr.size(0), size_1), nbr_2.view(nbr.size(0), size_2)], dim=1).flatten()
 99 |                 else:
100 |                     nbr = self.sampler(nbr, size_1).view(-1)
101 |                 num_nodes.append(nbr.size(0))
102 |                 h.append(x[nbr].to(device))
103 | 
104 |             for i, aggregator in enumerate(self.aggregators):
105 |                 self_x = h[:-1]
106 |                 neigh_x = []
107 |                 for j, n_x in enumerate(h[1:]):
108 |                     neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))
109 |                 out = self.snn[i](aggregator(self_x, neigh_x))
110 |                 if i != len(sizes) - 1:
111 |                     out = self.dropout(out)
112 |                     h = torch.split(out, num_nodes[:-(i + 1)])
113 | 
114 |             spikes.append(out)
115 | 
116 |         spikes = torch.stack(spikes, dim=1)  # [N,T,D]
117 |         if self.use_gs_delay:
118 |             spikes = self.delay(spikes)
119 |         self._last_spikes_for_loss = spikes
120 |         neuron.reset_net(self)
121 |         return spikes
122 | 
123 |     def forward(self, nodes):
124 |         spikes = self.encode(nodes)
125 |         if self.use_tsr:
126 |             return self.readout(spikes)
127 |         else:
128 |             return self.pooling(spikes.flatten(1))
129 | 
130 | 
131 | parser = argparse.ArgumentParser()
132 | parser.add_argument("--dataset", nargs="?", default="flickr",
133 |                     help="Datasets (Reddit and Flickr only). (default: Flickr)")
134 | parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2],
135 |                     help='Neighborhood sampling size for each layer. (default: [5, 2])')
136 | parser.add_argument('--hids', type=int, nargs='+',
137 |                     default=[512, 10], help='Hidden units for each layer. (default: [128, 10])')
138 | parser.add_argument("--aggr", nargs="?", default="mean",
139 |                     help="Aggregate function ('mean', 'sum'). (default: 'mean')")
140 | parser.add_argument("--sampler", nargs="?", default="sage",
141 |                     help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
142 | parser.add_argument("--surrogate", nargs="?", default="sigmoid",
143 |                     help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
144 | parser.add_argument("--neuron", nargs="?", default="LIF",
145 |                     help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
146 | parser.add_argument('--batch_size', type=int, default=2048,
147 |                     help='Batch size for training. (default: 1024)')
148 | parser.add_argument('--lr', type=float, default=5e-3,
149 |                     help='Learning rate for training. (default: 5e-3)')
150 | parser.add_argument('--alpha', type=float, default=1.0,
151 |                     help='Smooth factor for surrogate learning. (default: 1.0)')
152 | parser.add_argument('--T', type=int, default=15,
153 |                     help='Number of time steps. (default: 15)')
154 | parser.add_argument('--dropout', type=float, default=0.5,
155 |                     help='Dropout probability. (default: 0.5)')
156 | parser.add_argument('--epochs', type=int, default=100,
157 |                     help='Number of training epochs. (default: 100)')
158 | parser.add_argument('--concat', action='store_true',
159 |                     help='Whether to concat node representation and neighborhood representations. (default: False)')
160 | parser.add_argument('--seed', type=int, default=2022,
161 |                     help='Random seed for model. (default: 2022)')
162 | # 新增
163 | parser.add_argument('--use_gs_delay', type=int, default=1)
164 | parser.add_argument('--delay_groups', type=int, default=8)
165 | parser.add_argument('--delays', type=int, nargs='+', default=[1,3,5])
166 | parser.add_argument('--use_tsr', type=int, default=1)
167 | parser.add_argument('--spike_reg', type=float, default=0.0)
168 | parser.add_argument('--temp_reg', type=float, default=0.0)
169 | parser.add_argument('--alpha_warmup', type=float, default=0.3)
170 | 
171 | try:
172 |     args = parser.parse_args()
173 |     args.split_seed = 42
174 |     tab_printer(args)
175 | except:
176 |     parser.print_help()
177 |     exit(0)
178 | 
179 | assert len(args.hids) == len(args.sizes), "must be equal!"
180 | 
181 | root = "data/"  # Specify your root path
182 | 
183 | if args.dataset.lower() == "reddit":
184 |     dataset_obj = Reddit(osp.join(root, 'Reddit'))
185 |     data = dataset_obj[0]
186 | elif args.dataset.lower() == "flickr":
187 |     dataset_obj = Flickr(osp.join(root, 'Flickr'))
188 |     data = dataset_obj[0]
189 | 
190 | data.x = torch.nn.functional.normalize(data.x, dim=1)
191 | 
192 | set_seed(args.seed)
193 | 
194 | device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
195 | 
196 | y = data.y.to(device)
197 | 
198 | train_loader = DataLoader(data.train_mask.nonzero().view(-1), pin_memory=False, batch_size=args.batch_size, shuffle=True)
199 | val_loader = DataLoader(data.val_mask.nonzero().view(-1), pin_memory=False, batch_size=10000, shuffle=False)
200 | test_loader = DataLoader(data.test_mask.nonzero().view(-1), pin_memory=False, batch_size=10000, shuffle=False)
201 | 
202 | model = SpikeNet(dataset_obj.num_features, dataset_obj.num_classes, alpha=args.alpha,
203 |                  dropout=args.dropout, sampler=args.sampler, T=args.T,
204 |                  aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
205 |                  hids=args.hids, act=args.neuron, bias=True,
206 |                  use_gs_delay=bool(args.use_gs_delay), delay_groups=args.delay_groups,
207 |                  delay_set=tuple(args.delays), use_tsr=bool(args.use_tsr)).to(device)
208 | 
209 | optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
210 | loss_fn = nn.CrossEntropyLoss()
211 | 
212 | 
213 | def set_alpha_with_warmup(net: nn.Module, base_alpha: float, epoch: int, total_epochs: int, warmup_ratio: float):
214 |     if warmup_ratio <= 0:
215 |         cur_alpha = base_alpha
216 |     else:
217 |         warm_epochs = max(1, int(total_epochs * warmup_ratio))
218 |         factor = min(1.0, float(epoch) / float(warm_epochs))
219 |         cur_alpha = max(1e-3, base_alpha * factor)
220 |     for m in net.modules():
221 |         if hasattr(m, 'alpha') and isinstance(m.alpha, torch.Tensor):
222 |             m.alpha.data.fill_(cur_alpha)
223 | 
224 | 
225 | def train():
226 |     model.train()
227 |     for nodes in tqdm(train_loader, desc='Training'):
228 |         optimizer.zero_grad()
229 |         logits = model(nodes)
230 |         ce = loss_fn(logits, y[nodes])
231 |         loss = ce
232 |         spikes = model._last_spikes_for_loss
233 |         if spikes is not None:
234 |             if args.spike_reg > 0:
235 |                 loss = loss + args.spike_reg * spikes.mean()
236 |             if args.temp_reg > 0 and spikes.size(1) > 1:
237 |                 loss = loss + args.temp_reg * (spikes[:,1:] - spikes[:,:-1]).pow(2).mean()
238 |         loss.backward()
239 |         optimizer.step()
240 | 
241 | 
242 | @torch.no_grad()
243 | def test(loader):
244 |     model.eval()
245 |     logits = []
246 |     labels = []
247 |     for nodes in loader:
248 |         logits.append(model(nodes))
249 |         labels.append(y[nodes])
250 |     logits = torch.cat(logits, dim=0).cpu()
251 |     labels = torch.cat(labels, dim=0).cpu()
252 |     preds = logits.argmax(1)
253 |     metric_macro = metrics.f1_score(labels, preds, average='macro')
254 |     metric_micro = metrics.f1_score(labels, preds, average='micro')
255 |     return metric_macro, metric_micro
256 | 
257 | 
258 | best_val_metric = test_metric = 0
259 | start = time.time()
260 | for epoch in range(1, args.epochs + 1):
261 |     set_alpha_with_warmup(model, args.alpha, epoch, args.epochs, args.alpha_warmup)
262 |     train()
263 |     val_metric, test_metric = test(val_loader), test(test_loader)
264 |     if val_metric[1] > best_val_metric:
265 |         best_val_metric = val_metric[1]
266 |         best_test_metric = test_metric
267 |     end = time.time()
268 |     print(
269 |         f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s')
270 | 
271 | # # 保存脉冲嵌入（可选）
272 | # # emb = model.encode(torch.arange(data.num_nodes)).cpu()
273 | # # torch.save(emb, 'emb.pth')
```

## File: F:\SomeProjects\SpikeGNNNet2\project.md

- Extension: .md
- Language: markdown
- Size: 17554 bytes
- Created: 2025-09-19 01:33:43
- Modified: 2025-09-19 01:33:47

### Code

```markdown
  1 | # 方案总览（方法名可用：**SpikeNet-LED**：**L**earnable **E**fficient **D**elays）
  2 | 
  3 | 核心由三块组成，可单独或组合使用：
  4 | 
  5 | 1) **GS-Delay**（**G**roup-**S**hared Sparse Delay Kernel，组共享稀疏延迟核）用**组共享 + 离散稀疏**的 1D 时间卷积近似 SOTA 的**高斯延迟核**，以**几乎不增加 FLOPs**的代价，显式建模“信息滞后/延迟影响”，并保持梯度稳定；比 SOTA 的逐通道高斯核**更省参/省算**。灵感来自对方的延迟卷积思想与稳定性条件，我们保留其**理论动机**但在结构上做**效率友好的蒸馏**与离散化近似（见“原理”与“代码意见”）。
  6 | 2) **L2S-Gate**（**L**earnable **L**ocal **2**-source Sampling Gate，可学习时序-结构采样门）将你代码里的常量 `p`（当前累计图 vs. 演化图邻采样比例）变为**超轻量可学习门控** `p_{l,t}`（按层/按时刻），依据**当前时间片统计（新增边占比/平均度/平均放电率）**自动分配“取多少历史/多少当下”。**不对采样操作反传**，因此几乎**零额外开销**，却能明显提升鲁棒性与泛化。
  7 | 3) **TSR-Readout**（Temporal Separable Readout，可分离时序读出）
  8 |    用**Depthwise(时间维) 1D 卷积 + 1×1** 代替 `Linear(T·D, C)` 的大读出层，使**参数/T 解耦**，并显式建模时间模式；与 1) 兼容，可以直接把 GS-Delay 的稀疏核**复用在读出**或作为前置时间滤波。SOTA 里已证明**训练出的时间加权优于简单拼接**，我们给出**更高效的结构化替代**。
  9 | 
 10 | **训练层面的两处“低成本稳态增强”**（可选）：
 11 | 
 12 | - **SCAT**：自适应阈值 + 替代梯度课程（逐步增大 `alpha`），配合**平均放电率正则**，让梯度更稳、能耗受控；
 13 | - **Temporal Consistency**：相邻时刻表示一致性（L2/InfoNCE 的轻量变体），呼应 SOTA 的**时间平滑正则**但略微更强。
 14 | 
 15 | ---
 16 | 
 17 | ## 原理：为什么这三件事会有效？
 18 | 
 19 | ### （1）组共享稀疏延迟核（GS-Delay）
 20 | 
 21 | - **效仿但不复制 SOTA**：SOTA 用**高斯核**对历史脉冲加权并“推迟”到将来，缓解信息遗忘并拟合真实传播时延；同时给出了**σ 与核长 Ks 的约束**以避免梯度爆炸/消失，解释了为何延迟在训练上是安全的。
 22 | - **我们的改进**：把“每通道可学习高斯核”替换为**组共享（G 组）+ 离散稀疏**的延迟核：仅在有限延迟集合 Δ={1,3,5,…} 上放非零权，且**每组共享同一组核参数**。这样：
 23 |   - 复杂度从 **O(D·Ks)** → **O(G·|Δ|)**（D≫G, |Δ|≪Ks），**省参/省算**；
 24 |   - 离散核的**L1 归一 + 最大延迟≤Ks** 可直接承接 SOTA 的梯度有界推理思路（把高斯上界替换为离散权的上界），**稳定性不减**且便于实现。
 25 | - **与 SNN 的契合**：脉冲稀疏 → 实际有效卷积更少；离散核在稀疏脉冲上更易“事件驱动”地生效，提升**能效/速度**。
 26 | 
 27 | ### （2）可学习采样门（L2S-Gate）
 28 | 
 29 | - **动机**：你已有 `p` 在**累计图**与**演化图**采样之间分配邻居数（一个很棒的时序结构融合点）；不同层/时刻最佳 `p` 不同。
 30 | - **做法**：用 **σ(w·z_t+b)** 产出 `p_{l,t}`（`z_t` 为无梯度时间片统计：新增边比例、平均度、上一轮放电率均值等）。只改变**邻居配额**，**不改变采样算子/不反传**，训练开销≈0。
 31 | - **收益**：当**突发变化**或**度异质**时，模型自动“多看当前/多看历史”，降低方差、提升稳定性。与 SOTA 的“延迟使历史影响后效”形成**互补**：我们不仅延迟历史，还**自适应决定看多少历史**。
 32 | 
 33 | ### （3）可分离时序读出（TSR-Readout）
 34 | 
 35 | - **动机**：原始 `Linear(T·D,C)` 对 T 敏感、参数大；SOTA 用**可训练时间权重池化**证明“学到的时间权重更好”。
 36 | - **做法**：Depthwise 1D（仅时间维）+ Pointwise 1×1，再全局池化 → `Linear(D,C)`。
 37 | - **收益**：参数与 T 基本解耦，捕捉**短期与多尺度模式**；与 GS-Delay 共用 1D 卷积框架，工程上简单。
 38 | 
 39 | ### （4）训练稳态增强（SCAT + Temporal Consistency）
 40 | 
 41 | - **SCAT**：把 `gamma, thresh_decay` 变为**可学习且有界**的参数；`alpha` 采用**课程**（前期小、后期大）；再加**平均放电率正则**，可控能耗与梯度噪声。
 42 | - **时间一致性**：沿着 SOTA 的**相邻时刻平滑正则**，我们可用更轻的**投影+L2/InfoNCE**，提升低标注/分布漂移时段的鲁棒性，开销很小。
 43 | 
 44 | ---
 45 | 
 46 | # 与你仓库的对接（**只需小改动，多为“增量模块”**）
 47 | 
 48 | ## 需要新增的文件/模块
 49 | 
 50 | - `spikenet/temporal.py`（新）：放 **GS-Delay** 与 **TSR-Readout** 和一个**小工具函数**。
 51 | - （可选）`spikenet/gates.py`（新）：放 **L2S-Gate**（也可直接写进 `SpikeNet`）。
 52 | 
 53 | ## 需要小改动的现有文件
 54 | 
 55 | 1. **采样 C++ 小修（强烈建议）**：`spikenet/sample_neighber.cpp`你当前 `replace==true` 且 `row_count>num_neighbors` 时逻辑更接近“无放回”，会带来采样偏差。改为真正**有放回**采样（每次 `rand()%row_count` 取一个）即可；代价为常数级，能使实验更公允。
 56 | 2. **`main.py` / `main_static.py`**：
 57 |    - 在 `SpikeNet.__init__` 中注入 **TSR-Readout**（替换 `self.pooling`），注入 **GS-Delay**（在 `encode` 函数收集完 `spikes` 后做时间卷积）；
 58 |    - 打开 **L2S-Gate**：把常量 `self.p` 替换为 `p_{l,t}`；
 59 |    - 训练循环中加入 **放电率正则** 和（可选）**时间一致性**项；
 60 |    - （可选）用**线性 warm-up** 更新 `alpha`；
 61 | 3. **`spikenet/neuron.py`**：
 62 |    - 把全局常量 `gamma, thresh_decay` 变为**`nn.Parameter` + Sigmoid 约束**（保证(0,1)）；
 63 |    - 保持向后兼容（不改默认行为）。
 64 | 
 65 | ---
 66 | 
 67 | ## 关键代码建议（**骨架级**，便于你交给 AI 直接补全）
 68 | 
 69 | ### A) `spikenet/temporal.py`
 70 | 
 71 | ```python
 72 | import torch, torch.nn as nn
 73 | import torch.nn.functional as F
 74 | 
 75 | class GroupSparseDelay1D(nn.Module):
 76 |     """
 77 |     组共享 + 稀疏离散延迟核（仅沿时间维做 Depthwise 1D）
 78 |     输入: x [B, T, D] ；输出: y [B, T, D]
 79 |     """
 80 |     def __init__(self, D, T, groups=8, delays=(1,3,5), init='gaussian_like'):
 81 |         super().__init__()
 82 |         assert D % groups == 0
 83 |         self.D, self.T, self.G = D, T, groups
 84 |         self.delays = list(delays)  # 稀疏位置
 85 |         # 每组对每个 delay 一个权重（标量），共享到组内所有通道
 86 |         self.weight = nn.Parameter(torch.zeros(self.G, len(self.delays)))
 87 |         # 可选：中心/σ 的软约束参数，用于“高斯样式”初始化 + 稳定性约束
 88 |         self.register_buffer('mask', self._build_mask(T, self.delays))  # [len(delays), T] 的稀疏移位掩码
 89 |         self.reset_parameters(init)
 90 | 
 91 |     def _build_mask(self, T, delays):
 92 |         # 在时间维做“延迟移位”的 one-hot 卷积核集合（按延迟把1放在对应位置）
 93 |         m = torch.zeros(len(delays), T)
 94 |         for i, d in enumerate(delays):
 95 |             if d < T: m[i, -1-d] = 1.0   # 对齐“右端为当前时刻”
 96 |         return m  # 不参与学习
 97 | 
 98 |     def reset_parameters(self, init):
 99 |         nn.init.normal_(self.weight, mean=0, std=0.02)
100 |         if init == 'gaussian_like':
101 |             # 可按 SOTA 的 σ–Ks 启发，初始化靠近“温和扩散”的形态（稳定起步）  
102 |             with torch.no_grad():
103 |                 self.weight.add_(0.01)
104 | 
105 |     @torch.no_grad()
106 |     def _stability_projection(self):
107 |         # 简单稳定性约束：组内 L1 归一 & 权重幅度裁剪，等价于控制核能量与最大延迟贡献
108 |         w = self.weight.clamp_(-1.0, 1.0)
109 |         w.abs_().div_(w.abs().sum(dim=1, keepdim=True) + 1e-6)  # L1 归一（可改为 softmax）
110 |         self.weight.copy_(torch.sign(self.weight) * w)
111 | 
112 |     def forward(self, x):
113 |         # x: [B, T, D]  → reshape 为 [B*G, T, D/G] 逐组共享
114 |         B, T, D = x.shape
115 |         gC = D // self.G
116 |         xg = x.view(B, T, self.G, gC).permute(0,2,1,3)  # [B, G, T, gC]
117 |         # 组共享稀疏核：对每组，把 delays 的权重与 mask 做加权叠加，实现“稀疏时间卷积”
118 |         # 等价实现：用 F.conv1d 的 depthwise，提前把稀疏核拼成 [G,1,T]，也行
119 |         # 这里给出向量化思路：y_t = sum_d w_gd * x_{t-d}
120 |         M = self.mask.to(xg)                             # [K,T]
121 |         W = self.weight.softmax(dim=-1).unsqueeze(-1)    # [G,K,1]
122 |         # 收集延迟后的信号
123 |         xs = []
124 |         for i, d in enumerate(self.delays):
125 |             shifted = F.pad(xg, (0,0, d,0))[:, :, :T, :]  # 左 pad d，截到 T
126 |             xs.append(shifted.unsqueeze(2))               # [B,G,1,T,gC]
127 |         Xstk = torch.cat(xs, dim=2)                      # [B,G,K,T,gC]
128 |         y = (W.unsqueeze(0).unsqueeze(-1) * Xstk).sum(dim=2)  # [B,G,T,gC]
129 |         y = y.permute(0,2,1,3).contiguous().view(B, T, D)     # [B,T,D]
130 |         return y
131 | 
132 | class TemporalSeparableReadout(nn.Module):
133 |     """
134 |     Depthwise(时间) + Pointwise(1×1) + 池化 + FC
135 |     输入 [B, T, D] → 输出 [B, C]
136 |     """
137 |     def __init__(self, D, C, k=5):
138 |         super().__init__()
139 |         self.dw = nn.Conv1d(D, D, kernel_size=k, padding=k//2, groups=D, bias=False)
140 |         self.pw = nn.Conv1d(D, D, kernel_size=1, bias=False)
141 |         self.fc = nn.Linear(D, C)
142 | 
143 |     def forward(self, spikes):
144 |         # spikes: [B, T, D]
145 |         x = spikes.transpose(1, 2)     # [B, D, T]
146 |         x = self.pw(self.dw(x))        # [B, D, T]
147 |         x = x.mean(dim=-1)             # [B, D]
148 |         return self.fc(x)
149 | ```
150 | 
151 | ### B) 在 `main.py` 的最小接入（关键片段）
152 | 
153 | ```python
154 | from spikenet.temporal import GroupSparseDelay1D, TemporalSeparableReadout
155 | 
156 | class SpikeNet(nn.Module):
157 |     def __init__(..., hids=[128,10], sizes=[5,2], ..., concat=False, act='LIF',
158 |                  use_gs_delay=True, delay_groups=8, delay_set=(1,3,5),
159 |                  use_learnable_p=True, use_tsr=True):
160 |         super().__init__()
161 |         # ... 保留原聚合器与 SNN ...
162 |         self.sizes = sizes
163 |         self.use_gs_delay = use_gs_delay
164 |         self.use_tsr = use_tsr
165 |         self.use_learnable_p = use_learnable_p
166 | 
167 |         # 可学习 p 门（每层一个标量，也可做“含时刻上下文”的小门）
168 |         if self.use_learnable_p:
169 |             self.p_gate = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in sizes])
170 | 
171 |         # 计算最后一层通道数（concat 决定）
172 |         last_dim = (hids[-1]*2 if concat else hids[-1])
173 |         T = len(data)  # 动态图时间步
174 |         if self.use_gs_delay:
175 |             self.delay = GroupSparseDelay1D(D=last_dim, T=T, groups=delay_groups, delays=delay_set)
176 | 
177 |         if self.use_tsr:
178 |             self.readout = TemporalSeparableReadout(D=last_dim, C=out_features, k=5)
179 |         else:
180 |             self.pooling = nn.Linear(T * last_dim, out_features)
181 | 
182 |     def encode(self, nodes):
183 |         spikes = []
184 |         for t in range(len(data)):
185 |             # …… 原采样逻辑 ……
186 |             # 替换 p：
187 |             p_t = torch.sigmoid(self.p_gate[0]).item() if self.use_learnable_p else self.p
188 |             # 对于多层 sizes，可在每层里单独用 gate；这里演示第一层
189 |             # size1 = max(int(size * p_t), 1); size2 = size - size1
190 |             # …… 原图/演化图邻采样 & SAGE & SNN ……
191 |             spikes.append(out)  # [N, D]，最后一层脉冲
192 |         spikes = torch.stack(spikes, dim=1)  # [N, T, D]
193 |         if self.use_gs_delay:
194 |             spikes = self.delay(spikes)      # [N, T, D]
195 |         neuron.reset_net(self)
196 |         return spikes
197 | 
198 |     def forward(self, nodes):
199 |         spikes = self.encode(nodes)            # [B, T, D]
200 |         if self.use_tsr:
201 |             return self.readout(spikes)
202 |         else:
203 |             return self.pooling(spikes.flatten(1))
204 | ```
205 | 
206 | ### C) 训练处的轻量正则（`main.py`）
207 | 
208 | ```python
209 | # loss_fn = nn.CrossEntropyLoss()
210 | lambda_spike = 1e-4   # 放电率正则
211 | lambda_temp  = 1e-4   # 时间一致性（或 L2 平滑）
212 | 
213 | def train():
214 |     model.train()
215 |     for nodes in train_loader:
216 |         optimizer.zero_grad()
217 |         logits = model(nodes)                 # 前向里已做 delay/tsr
218 |         ce = loss_fn(logits, y[nodes])
219 | 
220 |         # 取 encode 的中间产物可通过返回“附加信息”或保存到 model 缓存；示例假设 model 缓存了最近一次 spikes
221 |         spikes = model._last_spikes_for_loss  # [B,T,D]（可按需存）
222 |         reg_spike = spikes.mean()             # 控制平均放电率
223 |         reg_temp  = (spikes[:,1:]-spikes[:,:-1]).pow(2).mean()  # 时间一致性（L2）
224 | 
225 |         loss = ce + lambda_spike*reg_spike + lambda_temp*reg_temp
226 |         loss.backward()
227 |         optimizer.step()
228 | ```
229 | 
230 | > 说明：若不想改动 `forward` 的返回结构，可在 `encode` 里把 `spikes` 缓存到 `self._last_spikes_for_loss`。
231 | 
232 | ### D) `neuron.py` 的微改（可选）
233 | 
234 | - 把 `gamma, thresh_decay` 变 `nn.Parameter` 并通过 `torch.sigmoid` 限制在 (0,1)，默认初始化回到原值；
235 | - 训练循环加一个**线性 warm-up** 更新 `alpha`（在 `args.epochs` 前 30% 线性增大到目标值），这是**常见 SNN 训练技巧**，与 SOTA 的稳定性分析一致方向。
236 | 
237 | ---
238 | 
239 | # 复杂度与资源（定性/可在论文中定量化）
240 | 
241 | - **参数量**：
242 |   - GS-Delay：从 O(D·Ks) → O(G·|Δ|)。典型 `D=128, Ks=9, G=8, |Δ|=3`，参数削减约一个数量级；
243 |   - TSR：读出层从 `T·D·C` → `D·(k + 1 + C)`，当 `T` 很大（如 Tmall）节省显著。
244 | - **FLOPs/时间**：
245 |   - Depthwise 1D 卷积与稀疏延迟移位开销很小；
246 |   - L2S-Gate 只涉及标量门控，不引入额外大算子；
247 |   - 正则项与课程学习开销可忽略。
248 | - **内存**：
249 |   - 读出层参数显著下降；
250 |   - 不改变 batch 的节点采样路径与张量规模。
251 | 
252 | ---
253 | 
254 | # 实验计划（最小可验证集 → 可写成主表 + 消融 + 效率表）
255 | 
256 | **数据**：DBLP/Tmall/Patent（与原设定一致，含 `merge step`）。**主表**：
257 | 
258 | - Baseline：原 SpikeNet；
259 | - +L2S；+TSR；+GS-Delay；三者组合（Ours）；
260 | - 复现/引用 SOTA（可按文献报告值或你们复现值，强调我们**更高效**）。**指标**：Macro/Micro-F1、单 epoch 时间、显存峰值、参数/FLOPs。**消融**：
261 | - 组数 G、延迟集合 Δ 的大小；
262 | - 是否加时间一致性/放电率正则；
263 | - 用 TSR vs. 原 FC 读出。**可视化**：
264 | - 学到的 `p_{l,t}` 热力图（门控如何在突发时间片提高对“当前图”的权重）；
265 | - 不同数据集上**延迟权重分布**（反映不同领域的时延差异；呼应 SOTA 的可解释性叙述）。
266 | 
267 | ---
268 | 
269 | # 论文写作结构与创新性亮点
270 | 
271 | 1. **问题与挑战**：动态图中“延迟效应 + 历史遗忘 + 大 T 下读出臃肿”。
272 | 2. **方法**：
273 |    - **GS-Delay**：以**组共享稀疏核**近似高斯延迟，**理论上**给出简单稳定性充分条件（L1 归一＋最大延迟约束 ⇒ 梯度有界），与 SOTA 的 σ–Ks 条件同向，**但更高效**；
274 |    - **L2S-Gate**：时序-结构采样配比的**可学习门控**，零额外算子，适配突发变化；
275 |    - **TSR-Readout**：**可分离时间卷积读出**，在 T 大时**降参显著**；
276 |    - **SCAT/Consistency**：稳态训练与鲁棒性增强（两行损失即可）。
277 | 3. **复杂度分析**：参数与 FLOPs 对比公式 + 表；
278 | 4. **实验**：三数据集 + 大 T 情况占优；
279 | 5. **可解释性**：延迟权重、门控 `p_{l,t}` 随时间/层的变化。
280 | 6. **结论**：**等效或更优精度 + 更高效率**，能在大规模动态图上稳定工作。
281 | 
282 | ---
283 | 
284 | ## 你可以让 AI 直接做的开发任务清单
285 | 
286 | - 新增 `spikenet/temporal.py`，按上述骨架补齐（含 mask 构造的更高效实现，如一次性组装 depthwise kernel 用 `Conv1d`）。
287 | - 在 `main.py` 中：
288 |   - 替换 `pooling` → `TemporalSeparableReadout`；
289 |   - `encode` 中插入 `GroupSparseDelay1D`；
290 |   - 将 `self.p` 改为 `p_{l,t}`（实现门控与时间统计的收集）；
291 |   - 训练里加入两项正则与 `alpha` 课程（warm-up）。
292 | - 修复 C++ 采样 `replace` 分支。
293 | - 提供 `--use_gs_delay/--use_tsr/--use_learnable_p` 的命令行开关；默认全开。
294 | - 写三个脚本：`run_dblp.sh / run_tmall.sh / run_patent.sh`（包含 ours 与各消融）。
295 | - 导出报告表格（F1、时间、显存、参数/FLOPs），生成论文表格占位符。
296 | 
297 | ---
298 | 
299 | ## 与 SOTA 的关系与差异化声明（写在论文 related/method 末尾）
300 | 
301 | - 我们**承认延迟建模的重要性**，并参考其稳定性分析；不同点是我们用**组共享稀疏核**与**可分离读出**达到**近似甚至更好的效果**，同时显著**降低参数与计算**；
302 | - 另外，我们把动态图中的“**历史 vs. 当前**”比例作为**可学习门控**，是对延迟思想的**结构域补强**，与 SOTA 互补。
303 | 
304 | ---
305 | 
306 | 如果你要更细的**补丁式代码改动**（直接对你贴出的文件逐行定位修改），我也可以按上述骨架把每个改动的具体行号与替换块列出来；你把这份方案丢给代码生成 AI，就能非常顺畅地落地实现与跑通实验。
```

## File: F:\SomeProjects\SpikeGNNNet2\readme.md

- Extension: .md
- Language: markdown
- Size: 7156 bytes
- Created: 2025-09-19 03:18:19
- Modified: 2023-09-27 17:42:24

### Code

```markdown
  1 | # SpikeNet
  2 | > [**Scaling Up Dynamic Graph Representation Learning via Spiking Neural Networks**](https://arxiv.org/abs/2208.10364) (AAAI 2023)
  3 | >
  4 | > Jintang Li, Zhouxin Yu, Zulun Zhu, Liang Chen, Qi Yu, Zibin Zheng, Sheng Tian, Ruofan Wu, Changhua Meng    
  5 | 
  6 | **This repository is an official PyTorch implementation of SpikeNet.**
  7 | <div align="center">
  8 |   <img src="figs/spikenet.png"/>
  9 | </div><br/>
 10 | 
 11 | 
 12 | # Abstract
 13 | Recent years have seen a surge in research on dynamic graph representation learning, which aims to model temporal graphs that are dynamic and evolving constantly over time. However, current work typically models graph dynamics with recurrent neural networks (RNNs), making them suffer seriously from computation and memory overheads on large temporal graphs. So far, scalability of dynamic graph representation learning on large temporal graphs remains one of the major challenges. In this paper, we present a scalable framework, namely SpikeNet, to efficiently capture the temporal and structural patterns of temporal graphs. We explore a new direction in that we can capture the evolving dynamics of temporal graphs with spiking neural networks (SNNs) instead of RNNs. As a low-power alternative to RNNs, SNNs explicitly model graph dynamics as spike trains of neuron populations and enable spike-based propagation in an efficient way. Experiments on three large real-world temporal graph datasets demonstrate that SpikeNet outperforms strong baselines on the temporal node classification task with lower computational costs. Particularly, SpikeNet generalizes to a large temporal graph (2M nodes and 13M edges) with significantly fewer parameters and computation overheads.
 14 | 
 15 | # Dataset
 16 | ## Overview
 17 | |             | DBLP    | Tmall     | Patent     |
 18 | | ----------- | ------- | --------- | ---------- |
 19 | | #nodes      | 28,085  | 577,314   | 2,738,012 |
 20 | | #edges      | 236,894 | 4,807,545 | 13,960,811 |
 21 | | #time steps | 27      | 186       | 25         |
 22 | | #classes    | 10      | 5         | 6          |
 23 | 
 24 | ## Download datasets
 25 | + DBLP
 26 | + Tmall
 27 | + Patent
 28 |   
 29 | All dataset can be found at [Dropbox](https://www.dropbox.com/sh/palzyh5box1uc1v/AACSLHB7PChT-ruN-rksZTCYa?dl=0). 
 30 | You can download the datasets and put them in the folder `data/`, e.g., `data/dblp`.
 31 | 
 32 | ## (Optional) Re-generate node features via DeepWalk
 33 | Since these datasets have no associated node features, we have generated node features via unsupervised DeepWalk method (saved as `.npy` format). 
 34 | You can find them at [Dropbox](https://www.dropbox.com/sh/palzyh5box1uc1v/AACSLHB7PChT-ruN-rksZTCYa?dl=0) as well. 
 35 | Only `dblp.npy` is uploaded due to size limit of Dropbox. 
 36 | 
 37 | (Update) The generated node features for Tmall and Patent datasets have been shared through Aliyun Drive, and the link is as follows: https://www.aliyundrive.com/s/LH9qa9XZmXa. 
 38 | 
 39 | Note: Since Aliyun Drive does not support direct sharing of npy files, you will need to manually change the file extension `.txt` to `.npy` after downloading. 
 40 | 
 41 | 
 42 | We also provide the script to generate the node features. Alternatively, you can generate them on your end (this will take about minutes to hours):
 43 | 
 44 | ```bash
 45 | python generate_feature.py --dataset dblp
 46 | python generate_feature.py --dataset tmall --normalize
 47 | python generate_feature.py --dataset patent --normalize
 48 | ```
 49 | 
 50 | ## Overall file structure
 51 | ```bash
 52 | SpikeNet
 53 | ├── data
 54 | │   ├── dblp
 55 | │   │   ├── dblp.npy
 56 | │   │   ├── dblp.txt
 57 | │   │   └── node2label.txt
 58 | │   ├── tmall
 59 | │   │   ├── tmall.npy
 60 | │   │   └── tmall.txt
 61 | │   │   ├── node2label.txt
 62 | │   ├── patent
 63 | │   │   ├── patent_edges.json
 64 | │   │   ├── patent_nodes.json
 65 | │   │   └── patent.npy
 66 | ├── figs
 67 | │   └── spikenet.png
 68 | ├── spikenet
 69 | │   ├── dataset.py
 70 | │   ├── deepwalk.py
 71 | │   ├── layers.py
 72 | │   ├── neuron.py
 73 | │   ├── sample_neighber.cpp
 74 | │   └── utils.py
 75 | ├── generate_feature.py
 76 | ├── main.py
 77 | ├── main_static.py
 78 | ├── README.md
 79 | ├── setup.py
 80 | ```
 81 | # Requirements
 82 | 
 83 | ```
 84 | tqdm==4.59.0
 85 | scipy==1.5.2
 86 | texttable==1.6.2
 87 | torch==1.9.0
 88 | numpy==1.22.4
 89 | numba==0.56.4
 90 | scikit_learn==1.0
 91 | torch_cluster (optional, only for random walk sampler)
 92 | ```
 93 | In fact, the version of these packages does not have to be consistent to ours. For example, Pytorch 1.6~-1.12 should also work.
 94 | 
 95 | 
 96 | 
 97 | # Usage
 98 | 
 99 | ## Build neighborhood sampler
100 | ```bash
101 | python setup.py install
102 | ```
103 | 
104 | ## Run SpikeNet
105 | 
106 | ```bash
107 | # DBLP
108 | python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.4
109 | python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.6
110 | python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.8
111 | 
112 | # Tmall
113 | python main.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.4
114 | python main.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.6
115 | python main.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.8
116 | 
117 | # Patent
118 | python main.py --dataset patent --hids 512 10 --batch_size 2048 --p 1.0 --train_size 0.4
119 | python main.py --dataset patent --hids 512 10 --batch_size 2048 --p 1.0 --train_size 0.6
120 | python main.py --dataset patent --hids 512 10 --batch_size 2048 --p 0.5 --train_size 0.8
121 | ```
122 | 
123 | 
124 | # On the extention to stastic graphs
125 | Actually, SpikeNet is not only applicaple for temporal graphs, it is also straightforward to extend to stastic graphs by defining a time step hyperparameter $T$ manually.
126 | In this way, the sampled subgraph at each time step naturally form graph snapshot. We can use SpikeNet to capture the *evolving* dynamics of sampled subgraphs.
127 | Due to space limit, we did not discuss this part in our paper. However, we believe this is indeed necessary to show the effectiveness of our work.
128 | 
129 | 
130 | We provide a simple example for the usage on stastic graphs datasets `Flickr` and `Reddit` (be sure you have PyTorch Geometric installed):
131 | 
132 | ```bash
133 | # Flickr
134 | python main_static.py --dataset flickr --surrogate super
135 | 
136 | # Reddit
137 | python main_static.py --dataset reddit --surrogate super
138 | ```
139 | 
140 | We report Micro-F1 score and the results are as follows:
141 | 
142 | | Method     | Flickr      | Reddit      |
143 | | ---------- | ----------- | ----------- |
144 | | GCN        | 0.492±0.003 | 0.933±0.000 |
145 | | GraphSAGE  | 0.501±0.013 | 0.953±0.001 |
146 | | FastGCN    | 0.504±0.001 | 0.924±0.001 |
147 | | S-GCN      | 0.482±0.003 | 0.964±0.001 |
148 | | AS-GCN     | 0.504±0.002 | 0.958±0.001 |
149 | | ClusterGCN | 0.481±0.005 | 0.954±0.001 |
150 | | GraphSAINT | 0.511±0.001 | 0.966±0.001 |
151 | | SpikeNet   | 0.515±0.003 | 0.953±0.001 |
152 | 
153 | # Reference
154 | ```bibtex
155 | @inproceedings{li2023scaling,
156 |   author    = {Jintang Li and
157 |                Zhouxin Yu and
158 |                Zulun Zhu and
159 |                Liang Chen and
160 |                Qi Yu and
161 |                Zibin Zheng and
162 |                Sheng Tian and
163 |                Ruofan Wu and
164 |                Changhua Meng},
165 |   title     = {Scaling Up Dynamic Graph Representation Learning via Spiking Neural
166 |                Networks},
167 |   booktitle = {{AAAI}},
168 |   pages     = {8588--8596},
169 |   publisher = {{AAAI} Press},
170 |   year      = {2023}
171 | }
172 | ```
```

## File: F:\SomeProjects\SpikeGNNNet2\setup.py

- Extension: .py
- Language: python
- Size: 327 bytes
- Created: 2025-09-19 01:31:58
- Modified: 2023-09-27 17:42:24

### Code

```python
 1 | from setuptools import setup
 2 | from torch.utils.cpp_extension import BuildExtension, CppExtension
 3 | 
 4 | setup(
 5 |     name="sample_neighber",
 6 |     ext_modules=[
 7 |         CppExtension("sample_neighber", sources=["spikenet/sample_neighber.cpp"], extra_compile_args=['-g']),
 8 | 
 9 |     ],
10 |     cmdclass={
11 |         "build_ext": BuildExtension
12 |     }
13 | )
```

## File: F:\SomeProjects\SpikeGNNNet2\spikenet\dataset.py

- Extension: .py
- Language: python
- Size: 11862 bytes
- Created: 2025-09-19 01:31:58
- Modified: 2023-09-27 17:42:24

### Code

```python
  1 | import math
  2 | import os.path as osp
  3 | from collections import defaultdict, namedtuple
  4 | from typing import Optional
  5 | 
  6 | import numpy as np
  7 | import scipy.sparse as sp
  8 | import torch
  9 | from sklearn import preprocessing
 10 | from sklearn.model_selection import train_test_split
 11 | from sklearn.preprocessing import LabelEncoder
 12 | from tqdm import tqdm
 13 | 
 14 | Data = namedtuple('Data', ['x', 'edge_index'])
 15 | 
 16 | 
 17 | def standard_normalization(arr):
 18 |     n_steps, n_node, n_dim = arr.shape
 19 |     arr_norm = preprocessing.scale(np.reshape(arr, [n_steps, n_node * n_dim]), axis=1)
 20 |     arr_norm = np.reshape(arr_norm, [n_steps, n_node, n_dim])
 21 |     return arr_norm
 22 | 
 23 | 
 24 | def edges_to_adj(edges, num_nodes, undirected=True):
 25 |     row, col = edges
 26 |     data = np.ones(len(row))
 27 |     N = num_nodes
 28 |     adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
 29 |     if undirected:
 30 |         adj = adj.maximum(adj.T)
 31 |     adj[adj > 1] = 1
 32 |     return adj
 33 | 
 34 | 
 35 | class Dataset:
 36 |     def __init__(self, name=None, root="./data"):
 37 |         self.name = name
 38 |         self.root = root
 39 |         self.x = None
 40 |         self.y = None
 41 |         self.num_features = None
 42 |         self.adj = []
 43 |         self.adj_evolve = []
 44 |         self.edges = []
 45 |         self.edges_evolve = []
 46 | 
 47 |     def _read_feature(self):
 48 |         filename = osp.join(self.root, self.name, f"{self.name}.npy")
 49 |         if osp.exists(filename):
 50 |             return np.load(filename)
 51 |         else:
 52 |             return None
 53 | 
 54 |     def split_nodes(
 55 |         self,
 56 |         train_size: float = 0.4,
 57 |         val_size: float = 0.0,
 58 |         test_size: float = 0.6,
 59 |         random_state: Optional[int] = None,
 60 |     ):
 61 |         val_size = 0. if val_size is None else val_size
 62 |         assert train_size + val_size + test_size <= 1.0
 63 | 
 64 |         y = self.y
 65 |         train_nodes, test_nodes = train_test_split(
 66 |             torch.arange(y.size(0)),
 67 |             train_size=train_size + val_size,
 68 |             test_size=test_size,
 69 |             random_state=random_state,
 70 |             stratify=y)
 71 | 
 72 |         if val_size:
 73 |             train_nodes, val_nodes = train_test_split(
 74 |                 train_nodes,
 75 |                 train_size=train_size / (train_size + val_size),
 76 |                 random_state=random_state,
 77 |                 stratify=y[train_nodes])
 78 |         else:
 79 |             val_nodes = None
 80 | 
 81 |         self.train_nodes = train_nodes
 82 |         self.val_nodes = val_nodes
 83 |         self.test_nodes = test_nodes
 84 | 
 85 |     def split_edges(
 86 |         self,
 87 |         train_stamp: float = 0.7,
 88 |         train_size: float = None,
 89 |         val_size: float = 0.1,
 90 |         test_size: float = 0.2,
 91 |         random_state: int = None,
 92 |     ):
 93 | 
 94 |         if random_state is not None:
 95 |             torch.manual_seed(random_state)
 96 | 
 97 |         num_edges = self.edges[-1].size(-1)
 98 |         train_stamp = train_stamp if train_stamp >= 1 else math.ceil(len(self) * train_stamp)
 99 | 
100 |         train_edges = torch.LongTensor(np.hstack(self.edges_evolve[:train_stamp]))
101 |         if train_size is not None:
102 |             assert 0 < train_size < 1
103 |             num_train = math.floor(train_size * num_edges)
104 |             perm = torch.randperm(train_edges.size(1))[:num_train]
105 |             train_edges = train_edges[:, perm]
106 | 
107 |         num_val = math.floor(val_size * num_edges)
108 |         num_test = math.floor(test_size * num_edges)
109 |         testing_edges = torch.LongTensor(np.hstack(self.edges_evolve[train_stamp:]))
110 |         perm = torch.randperm(testing_edges.size(1))
111 | 
112 |         assert num_val + num_test <= testing_edges.size(1)
113 | 
114 |         self.train_stamp = train_stamp
115 |         self.train_edges = train_edges
116 |         self.val_edges = testing_edges[:, perm[:num_val]]
117 |         self.test_edges = testing_edges[:, perm[num_val:num_val + num_test]]
118 | 
119 |     def __getitem__(self, time_index: int):
120 |         x = self.x[time_index]
121 |         edge_index = self.edges[time_index]
122 |         snapshot = Data(x=x, edge_index=edge_index)
123 |         return snapshot
124 | 
125 |     def __next__(self):
126 |         if self.t < len(self):
127 |             snapshot = self.__getitem__(self.t)
128 |             self.t = self.t + 1
129 |             return snapshot
130 |         else:
131 |             self.t = 0
132 |             raise StopIteration
133 | 
134 |     def __iter__(self):
135 |         self.t = 0
136 |         return self
137 | 
138 |     def __len__(self):
139 |         return len(self.adj)
140 | 
141 |     def __repr__(self):
142 |         return self.name
143 | 
144 | 
145 | class DBLP(Dataset):
146 |     def __init__(self, root="./data", normalize=True):
147 |         super().__init__(name='dblp', root=root)
148 |         edges_evolve, self.num_nodes = self._read_graph()
149 |         x = self._read_feature()
150 |         y = self._read_label()
151 | 
152 |         if x is not None:
153 |             if normalize:
154 |                 x = standard_normalization(x)
155 |             self.num_features = x.shape[-1]
156 |             self.x = torch.FloatTensor(x)
157 | 
158 |         self.num_classes = y.max() + 1
159 | 
160 |         edges = [edges_evolve[0]]
161 |         for e_now in edges_evolve[1:]:
162 |             e_last = edges[-1]
163 |             edges.append(np.hstack([e_last, e_now]))
164 | 
165 |         self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
166 |         self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
167 |         self.edges = [torch.LongTensor(edge) for edge in edges]
168 |         self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately
169 | 
170 |         self.y = torch.LongTensor(y)
171 | 
172 |     def _read_graph(self):
173 |         filename = osp.join(self.root, self.name, f"{self.name}.txt")
174 |         d = defaultdict(list)
175 |         N = 0
176 |         with open(filename) as f:
177 |             for line in f:
178 |                 x, y, t = line.strip().split()
179 |                 x, y = int(x), int(y)
180 |                 d[t].append((x, y))
181 |                 N = max(N, x)
182 |                 N = max(N, y)
183 |         N += 1
184 |         edges = []
185 |         for time in sorted(d):
186 |             row, col = zip(*d[time])
187 |             edge_now = np.vstack([row, col])
188 |             edges.append(edge_now)
189 |         return edges, N
190 | 
191 |     def _read_label(self):
192 |         filename = osp.join(self.root, self.name, "node2label.txt")
193 |         nodes = []
194 |         labels = []
195 |         with open(filename) as f:
196 |             for line in f:
197 |                 node, label = line.strip().split()
198 |                 nodes.append(int(node))
199 |                 labels.append(label)
200 | 
201 |         nodes = np.array(nodes)
202 |         labels = LabelEncoder().fit_transform(labels)
203 | 
204 |         assert np.allclose(nodes, np.arange(nodes.size))
205 |         return labels
206 | 
207 | 
208 | def merge(edges, step=1):
209 |     if step == 1:
210 |         return edges
211 |     i = 0
212 |     length = len(edges)
213 |     out = []
214 |     while i < length:
215 |         e = edges[i:i + step]
216 |         if len(e):
217 |             out.append(np.hstack(e))
218 |         i += step
219 |     print(f'Edges has been merged from {len(edges)} timestamps to {len(out)} timestamps')
220 |     return out
221 | 
222 | 
223 | class Tmall(Dataset):
224 |     def __init__(self, root="./data", normalize=True):
225 |         super().__init__(name='tmall', root=root)
226 |         edges_evolve, self.num_nodes = self._read_graph()
227 |         x = self._read_feature()
228 | 
229 |         y, labeled_nodes = self._read_label()
230 |         # reindexing
231 |         others = set(range(self.num_nodes)) - set(labeled_nodes.tolist())
232 |         new_index = np.hstack([labeled_nodes, list(others)])
233 |         whole_nodes = np.arange(self.num_nodes)
234 |         mapping_dict = dict(zip(new_index, whole_nodes))
235 |         mapping = np.vectorize(mapping_dict.get)(whole_nodes)
236 |         edges_evolve = [mapping[e] for e in edges_evolve]
237 | 
238 |         edges_evolve = merge(edges_evolve, step=10)
239 | 
240 |         if x is not None:
241 |             if normalize:
242 |                 x = standard_normalization(x)
243 |             self.num_features = x.shape[-1]
244 |             self.x = torch.FloatTensor(x)
245 | 
246 |         self.num_classes = y.max() + 1
247 | 
248 |         edges = [edges_evolve[0]]
249 |         for e_now in edges_evolve[1:]:
250 |             e_last = edges[-1]
251 |             edges.append(np.hstack([e_last, e_now]))
252 | 
253 |         self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
254 |         self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
255 |         self.edges = [torch.LongTensor(edge) for edge in edges]
256 |         self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately
257 | 
258 |         self.mapping = mapping
259 |         self.y = torch.LongTensor(y)
260 | 
261 |     def _read_graph(self):
262 |         filename = osp.join(self.root, self.name, f"{self.name}.txt")
263 |         d = defaultdict(list)
264 |         N = 0
265 |         with open(filename) as f:
266 |             for line in tqdm(f, desc='loading edges'):
267 |                 x, y, t = line.strip().split()
268 |                 x, y = int(x), int(y)
269 |                 d[t].append((x, y))
270 |                 N = max(N, x)
271 |                 N = max(N, y)
272 |         N += 1
273 |         edges = []
274 |         for time in sorted(d):
275 |             row, col = zip(*d[time])
276 |             edge_now = np.vstack([row, col])
277 |             edges.append(edge_now)
278 |         return edges, N
279 | 
280 |     def _read_label(self):
281 |         filename = osp.join(self.root, self.name, "node2label.txt")
282 |         nodes = []
283 |         labels = []
284 |         with open(filename) as f:
285 |             for line in tqdm(f, desc='loading nodes'):
286 |                 node, label = line.strip().split()
287 |                 nodes.append(int(node))
288 |                 labels.append(label)
289 | 
290 |         labeled_nodes = np.array(nodes)
291 |         labels = LabelEncoder().fit_transform(labels)
292 |         return labels, labeled_nodes
293 | 
294 | 
295 | class Patent(Dataset):
296 |     def __init__(self, root="./data", normalize=True):
297 |         super().__init__(name='patent', root=root)
298 |         edges_evolve = self._read_graph()
299 |         y = self._read_label()
300 |         edges_evolve = merge(edges_evolve, step=2)
301 |         x = self._read_feature()
302 | 
303 |         if x is not None:
304 |             if normalize:
305 |                 x = standard_normalization(x)
306 |             self.num_features = x.shape[-1]
307 |             self.x = torch.FloatTensor(x)
308 | 
309 |         self.num_nodes = y.size
310 |         self.num_features = x.shape[-1]
311 |         self.num_classes = y.max() + 1
312 | 
313 |         edges = [edges_evolve[0]]
314 |         for e_now in edges_evolve[1:]:
315 |             e_last = edges[-1]
316 |             edges.append(np.hstack([e_last, e_now]))
317 | 
318 |         self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
319 |         self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
320 |         self.edges = [torch.LongTensor(edge) for edge in edges]
321 |         self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately
322 | 
323 |         self.x = torch.FloatTensor(x)
324 |         self.y = torch.LongTensor(y)
325 | 
326 |     def _read_graph(self):
327 |         filename = osp.join(self.root, self.name, f"{self.name}_edges.json")
328 |         time_edges = defaultdict(list)
329 |         with open(filename) as f:
330 |             for line in tqdm(f, desc='loading patent_edges'):
331 |                 # src nodeID, dst nodeID, date, src originalID, dst originalID
332 |                 src, dst, date, _, _ = eval(line)
333 |                 date = date // 1e4
334 |                 time_edges[date].append((src, dst))
335 | 
336 |         edges = []
337 |         for time in sorted(time_edges):
338 |             edges.append(np.transpose(time_edges[time]))
339 |         return edges
340 | 
341 |     def _read_label(self):
342 |         filename = osp.join(self.root, self.name, f"{self.name}_nodes.json")
343 |         labels = []
344 |         with open(filename) as f:
345 |             for line in tqdm(f, desc='loading patent_nodes'):
346 |                 # nodeID, originalID, date, node class
347 |                 node, _, date, label = eval(line)
348 |                 date = date // 1e4
349 |                 labels.append(label - 1)
350 |         labels = np.array(labels)
351 |         return labels
```

## File: F:\SomeProjects\SpikeGNNNet2\spikenet\deepwalk.py

- Extension: .py
- Language: python
- Size: 5290 bytes
- Created: 2025-09-19 01:31:58
- Modified: 2023-09-27 17:42:24

### Code

```python
  1 | from distutils.version import LooseVersion
  2 | 
  3 | import gensim
  4 | import numpy as np
  5 | import scipy.sparse as sp
  6 | from gensim.models import Word2Vec as _Word2Vec
  7 | from numba import njit
  8 | from sklearn import preprocessing
  9 | 
 10 | 
 11 | class DeepWalk:
 12 |     r"""Implementation of `"DeepWalk" <https://arxiv.org/abs/1403.6652>`_
 13 |     from the KDD '14 paper "DeepWalk: Online Learning of Social Representations".
 14 |     The procedure uses random walks to approximate the pointwise mutual information
 15 |     matrix obtained by pooling normalized adjacency matrix powers. This matrix
 16 |     is decomposed by an approximate factorization technique.
 17 |     """
 18 | 
 19 |     def __init__(self, dimensions: int = 64,
 20 |                  walk_length: int = 80,
 21 |                  walk_number: int = 10,
 22 |                  workers: int = 3,
 23 |                  window_size: int = 5,
 24 |                  epochs: int = 1,
 25 |                  learning_rate: float = 0.025,
 26 |                  negative: int = 1,
 27 |                  name: str = None,
 28 |                  seed: int = None):
 29 | 
 30 |         kwargs = locals()
 31 |         kwargs.pop("self")
 32 |         kwargs.pop("__class__", None)
 33 | 
 34 |         self.set_hyparas(kwargs)
 35 | 
 36 |     def set_hyparas(self, kwargs: dict):
 37 |         for k, v in kwargs.items():
 38 |             setattr(self, k, v)
 39 |         self.hyparas = kwargs
 40 | 
 41 |     def fit(self, graph: sp.csr_matrix):
 42 |         walks = RandomWalker(walk_length=self.walk_length,
 43 |                              walk_number=self.walk_number).walk(graph)
 44 |         sentences = [list(map(str, walk)) for walk in walks]
 45 |         model = Word2Vec(sentences,
 46 |                          sg=1,
 47 |                          hs=0,
 48 |                          alpha=self.learning_rate,
 49 |                          iter=self.epochs,
 50 |                          size=self.dimensions,
 51 |                          window=self.window_size,
 52 |                          workers=self.workers,
 53 |                          negative=self.negative,
 54 |                          seed=self.seed)
 55 |         self._embedding = model.get_embedding()
 56 | 
 57 |     def get_embedding(self, normalize=True) -> np.array:
 58 |         """Getting the node embedding."""
 59 |         embedding = self._embedding
 60 |         if normalize:
 61 |             embedding = preprocessing.normalize(embedding)
 62 |         return embedding
 63 | 
 64 | 
 65 | class RandomWalker:
 66 |     """Fast first-order random walks in DeepWalk
 67 | 
 68 |     Parameters:
 69 |     -----------
 70 |     walk_number (int): Number of random walks. Default is 10.
 71 |     walk_length (int): Length of random walks. Default is 80.
 72 |     """
 73 | 
 74 |     def __init__(self, walk_length: int = 80, walk_number: int = 10):
 75 |         self.walk_length = walk_length
 76 |         self.walk_number = walk_number
 77 | 
 78 |     def walk(self, graph: sp.csr_matrix):
 79 |         walks = self.random_walk(graph.indices,
 80 |                                  graph.indptr,
 81 |                                  walk_length=self.walk_length,
 82 |                                  walk_number=self.walk_number)
 83 |         return walks
 84 | 
 85 |     @staticmethod
 86 |     @njit(nogil=True)
 87 |     def random_walk(indices,
 88 |                     indptr,
 89 |                     walk_length,
 90 |                     walk_number):
 91 |         N = len(indptr) - 1
 92 |         for _ in range(walk_number):
 93 |             for n in range(N):
 94 |                 walk = [n]
 95 |                 current_node = n
 96 |                 for _ in range(walk_length - 1):
 97 |                     neighbors = indices[
 98 |                         indptr[current_node]:indptr[current_node + 1]]
 99 |                     if neighbors.size == 0:
100 |                         break
101 |                     current_node = np.random.choice(neighbors)
102 |                     walk.append(current_node)
103 | 
104 |                 yield walk
105 | 
106 | 
107 | class Word2Vec(_Word2Vec):
108 |     """A compatible version of Word2Vec"""
109 | 
110 |     def __init__(self, sentences=None, sg=0, hs=0, alpha=0.025, iter=5, size=100, window=5, workers=3, negative=5, seed=None, **kwargs):
111 |         if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
112 |             super().__init__(sentences,
113 |                              size=size,
114 |                              window=window,
115 |                              min_count=0,
116 |                              alpha=alpha,
117 |                              sg=sg,
118 |                              workers=workers,
119 |                              iter=iter,
120 |                              negative=negative,
121 |                              hs=hs,
122 |                              compute_loss=True,
123 |                              seed=seed, **kwargs)
124 | 
125 |         else:
126 |             super().__init__(sentences,
127 |                              vector_size=size,
128 |                              window=window,
129 |                              min_count=0,
130 |                              alpha=alpha,
131 |                              sg=sg,
132 |                              workers=workers,
133 |                              epochs=iter,
134 |                              negative=negative,
135 |                              hs=hs,
136 |                              compute_loss=True,
137 |                              seed=seed, **kwargs)
138 | 
139 |     def get_embedding(self):
140 |         if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
141 |             embedding = self.wv.vectors[np.fromiter(
142 |                 map(int, self.wv.index2word), np.int32).argsort()]
143 |         else:
144 |             embedding = self.wv.vectors[np.fromiter(
145 |                 map(int, self.wv.index_to_key), np.int32).argsort()]
146 | 
147 |         return embedding
```

## File: F:\SomeProjects\SpikeGNNNet2\spikenet\gates.py

- Extension: .py
- Language: python
- Size: 1251 bytes
- Created: 2025-09-19 03:14:25
- Modified: 2025-09-19 04:39:29

### Code

```python
 1 | # File: spikenet/gates.py
 2 | 
 3 | import torch
 4 | import torch.nn as nn
 5 | 
 6 | class L2SGate(nn.Module):
 7 |     """
 8 |     Learnable Local 2-source Sampling Gate（按层的极轻量门控），不对采样操作反传。
 9 |     输入: stats_t ∈ R^F（如: [新增边占比, 平均度]），输出: p_{l,t} ∈ (0,1) 每层一个。
10 |     """
11 |     def __init__(self, num_layers: int, in_features: int = 2, base_p: float = 0.5):
12 |         super().__init__()
13 |         self.num_layers = num_layers
14 |         self.base_p = base_p
15 |         self.w = nn.Parameter(torch.zeros(num_layers, in_features))
16 |         self.b = nn.Parameter(torch.zeros(num_layers))
17 | 
18 |     @torch.no_grad()
19 |     def forward(self, stats_t: torch.Tensor) -> torch.Tensor:
20 |         # ⚠️ 关键修复：把输入统计量搬到参数所在设备/精度
21 |         z = stats_t.detach().to(self.w.device, dtype=self.w.dtype).view(-1)  # [F]
22 |         logits = torch.mv(self.w, z) + self.b                               # [L]
23 |         p = torch.sigmoid(logits)                                           # (0,1)
24 |         base = torch.as_tensor(self.base_p, dtype=p.dtype, device=p.device)
25 |         p = 0.5 * p + 0.5 * base                                            # 温和融合
26 |         return p  # [L]
```

## File: F:\SomeProjects\SpikeGNNNet2\spikenet\layers.py

- Extension: .py
- Language: python
- Size: 1225 bytes
- Created: 2025-09-19 01:31:58
- Modified: 2023-09-27 17:42:24

### Code

```python
 1 | import torch
 2 | import torch.nn as nn
 3 | 
 4 | 
 5 | class SAGEAggregator(nn.Module):
 6 |     def __init__(self, in_features, out_features,
 7 |                  aggr='mean',
 8 |                  concat=False,
 9 |                  bias=False):
10 | 
11 |         super().__init__()
12 |         self.in_features = in_features
13 |         self.out_features = out_features
14 |         self.concat = concat
15 | 
16 |         self.aggr = aggr
17 |         self.aggregator = {'mean': torch.mean, 'sum': torch.sum}[aggr]
18 | 
19 |         self.lin_l = nn.Linear(in_features, out_features, bias=bias)
20 |         self.lin_r = nn.Linear(in_features, out_features, bias=bias)
21 | 
22 |     def forward(self, x, neigh_x):
23 |         if not isinstance(x, torch.Tensor):
24 |             x = torch.cat(x, dim=0)
25 | 
26 |         if not isinstance(neigh_x, torch.Tensor):
27 |             neigh_x = torch.cat([self.aggregator(h, dim=1)
28 |                                 for h in neigh_x], dim=0)
29 |         else:
30 |             neigh_x = self.aggregator(neigh_x, dim=1)
31 | 
32 |         x = self.lin_l(x)
33 |         neigh_x = self.lin_r(neigh_x)
34 |         out = torch.cat([x, neigh_x], dim=1) if self.concat else x + neigh_x
35 |         return out
36 | 
37 |     def __repr__(self):
38 |         return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, aggr={self.aggr})"
```

## File: F:\SomeProjects\SpikeGNNNet2\spikenet\neuron.py

- Extension: .py
- Language: python
- Size: 7039 bytes
- Created: 2025-09-19 01:31:58
- Modified: 2023-09-27 17:42:24

### Code

```python
  1 | from math import pi
  2 | 
  3 | import torch
  4 | import torch.nn as nn
  5 | 
  6 | gamma = 0.2
  7 | thresh_decay = 0.7
  8 | 
  9 | 
 10 | def reset_net(net: nn.Module):
 11 |     for m in net.modules():
 12 |         if hasattr(m, 'reset'):
 13 |             m.reset()
 14 | 
 15 | 
 16 | def heaviside(x: torch.Tensor):
 17 |     return x.ge(0)
 18 | 
 19 | 
 20 | def gaussian(x, mu, sigma):
 21 |     """
 22 |     Gaussian PDF with broadcasting.
 23 |     """
 24 |     return torch.exp(-((x - mu) * (x - mu)) / (2 * sigma * sigma)) / (sigma * torch.sqrt(2 * torch.tensor(pi)))
 25 | 
 26 | 
 27 | class BaseSpike(torch.autograd.Function):
 28 |     """
 29 |     Baseline spiking function.
 30 |     """
 31 | 
 32 |     @staticmethod
 33 |     def forward(ctx, x, alpha):
 34 |         ctx.save_for_backward(x, alpha)
 35 |         return x.gt(0).float()
 36 | 
 37 |     @staticmethod
 38 |     def backward(ctx, grad_output):
 39 |         raise NotImplementedError
 40 | 
 41 | 
 42 | class SuperSpike(BaseSpike):
 43 |     """
 44 |     Spike function with SuperSpike surrogate gradient from
 45 |     "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks", Zenke et al. 2018.
 46 | 
 47 |     Design choices:
 48 |     - Height of 1 ("The Remarkable Robustness of Surrogate Gradient...", Zenke et al. 2021)
 49 |     - alpha scaled by 10 ("Training Deep Spiking Neural Networks", Ledinauskas et al. 2020)
 50 |     """
 51 | 
 52 |     @staticmethod
 53 |     def backward(ctx, grad_output):
 54 |         x, alpha = ctx.saved_tensors
 55 |         grad_input = grad_output.clone()
 56 |         sg = 1 / (1 + alpha * x.abs()) ** 2
 57 |         return grad_input * sg, None
 58 | 
 59 | 
 60 | class MultiGaussSpike(BaseSpike):
 61 |     """
 62 |     Spike function with multi-Gaussian surrogate gradient from
 63 |     "Accurate and efficient time-domain classification...", Yin et al. 2021.
 64 | 
 65 |     Design choices:
 66 |     - Hyperparameters determined through grid search (Yin et al. 2021)
 67 |     """
 68 | 
 69 |     @staticmethod
 70 |     def backward(ctx, grad_output):
 71 |         x, alpha = ctx.saved_tensors
 72 |         grad_input = grad_output.clone()
 73 |         zero = torch.tensor(0.0)  # no need to specify device for 0-d tensors
 74 |         sg = (
 75 |             1.15 * gaussian(x, zero, alpha)
 76 |             - 0.15 * gaussian(x, alpha, 6 * alpha)
 77 |             - 0.15 * gaussian(x, -alpha, 6 * alpha)
 78 |         )
 79 |         return grad_input * sg, None
 80 | 
 81 | 
 82 | class TriangleSpike(BaseSpike):
 83 |     """
 84 |     Spike function with triangular surrogate gradient
 85 |     as in Bellec et al. 2020.
 86 |     """
 87 | 
 88 |     @staticmethod
 89 |     def backward(ctx, grad_output):
 90 |         x, alpha = ctx.saved_tensors
 91 |         grad_input = grad_output.clone()
 92 |         sg = torch.nn.functional.relu(1 - alpha * x.abs())
 93 |         return grad_input * sg, None
 94 | 
 95 | 
 96 | class ArctanSpike(BaseSpike):
 97 |     """
 98 |     Spike function with derivative of arctan surrogate gradient.
 99 |     Featured in Fang et al. 2020/2021.
100 |     """
101 | 
102 |     @staticmethod
103 |     def backward(ctx, grad_output):
104 |         x, alpha = ctx.saved_tensors
105 |         grad_input = grad_output.clone()
106 |         sg = 1 / (1 + alpha * x * x)
107 |         return grad_input * sg, None
108 | 
109 | 
110 | class SigmoidSpike(BaseSpike):
111 | 
112 |     @staticmethod
113 |     def backward(ctx, grad_output):
114 |         x, alpha = ctx.saved_tensors
115 |         grad_input = grad_output.clone()
116 |         sgax = (x * alpha).sigmoid_()
117 |         sg = (1. - sgax) * sgax * alpha
118 |         return grad_input * sg, None
119 | 
120 | 
121 | def superspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
122 |     return SuperSpike.apply(x - thresh, alpha)
123 | 
124 | 
125 | def mgspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(0.5)):
126 |     return MultiGaussSpike.apply(x - thresh, alpha)
127 | 
128 | 
129 | def sigmoidspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
130 |     return SigmoidSpike.apply(x - thresh, alpha)
131 | 
132 | 
133 | def trianglespike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
134 |     return TriangleSpike.apply(x - thresh, alpha)
135 | 
136 | 
137 | def arctanspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
138 |     return ArctanSpike.apply(x - thresh, alpha)
139 | 
140 | 
141 | SURROGATE = {'sigmoid': sigmoidspike, 'triangle': trianglespike, 'arctan': arctanspike,
142 |              'mg': mgspike, 'super': superspike}
143 | 
144 | 
145 | class IF(nn.Module):
146 |     def __init__(self, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle'):
147 |         super().__init__()
148 |         self.v_threshold = v_threshold
149 |         self.v_reset = v_reset
150 |         self.surrogate = SURROGATE.get(surrogate)
151 |         self.register_buffer("alpha", torch.as_tensor(
152 |             alpha, dtype=torch.float32))
153 |         self.reset()
154 | 
155 |     def reset(self):
156 |         self.v = 0.
157 |         self.v_th = self.v_threshold
158 | 
159 |     def forward(self, dv):
160 |         # 1. charge
161 |         self.v += dv
162 |         # 2. fire
163 |         spike = self.surrogate(self.v, self.v_threshold, self.alpha)
164 |         # 3. reset
165 |         self.v = (1 - spike) * self.v + spike * self.v_reset
166 |         # 4. threhold updates
167 |         # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
168 |         self.v_th = gamma * spike + self.v_th * thresh_decay
169 |         return spike
170 | 
171 | 
172 | class LIF(nn.Module):
173 |     def __init__(self, tau=1.0, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle'):
174 |         super().__init__()
175 |         self.v_threshold = v_threshold
176 |         self.v_reset = v_reset
177 |         self.surrogate = SURROGATE.get(surrogate)
178 |         self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
179 |         self.register_buffer("alpha", torch.as_tensor(
180 |             alpha, dtype=torch.float32))
181 |         self.reset()
182 | 
183 |     def reset(self):
184 |         self.v = 0.
185 |         self.v_th = self.v_threshold
186 | 
187 |     def forward(self, dv):
188 |         # 1. charge
189 |         self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
190 |         # 2. fire
191 |         spike = self.surrogate(self.v, self.v_th, self.alpha)
192 |         # 3. reset
193 |         self.v = (1 - spike) * self.v + spike * self.v_reset
194 |         # 4. threhold updates
195 |         # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
196 |         self.v_th = gamma * spike + self.v_th * thresh_decay
197 |         return spike
198 | 
199 | 
200 | class PLIF(nn.Module):
201 |     def __init__(self, tau=1.0, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle'):
202 |         super().__init__()
203 |         self.v_threshold = v_threshold
204 |         self.v_reset = v_reset
205 |         self.surrogate = SURROGATE.get(surrogate)
206 |         self.register_parameter("tau", nn.Parameter(
207 |             torch.as_tensor(tau, dtype=torch.float32)))
208 |         self.register_buffer("alpha", torch.as_tensor(
209 |             alpha, dtype=torch.float32))
210 |         self.reset()
211 | 
212 |     def reset(self):
213 |         self.v = 0.
214 |         self.v_th = self.v_threshold
215 | 
216 |     def forward(self, dv):
217 |         # 1. charge
218 |         self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
219 |         # 2. fire
220 |         spike = self.surrogate(self.v, self.v_th, self.alpha)
221 |         # 3. reset
222 |         self.v = (1 - spike) * self.v + spike * self.v_reset
223 |         # 4. threhold updates
224 |         # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
225 |         self.v_th = gamma * spike + self.v_th * thresh_decay
226 |         return spike
```

## File: F:\SomeProjects\SpikeGNNNet2\spikenet\sample_neighber.cpp

- Extension: .cpp
- Language: cpp
- Size: 3824 bytes
- Created: 2025-09-19 01:31:58
- Modified: 2025-09-19 04:11:20

### Code

```cpp
  1 | // File: spikenet/sample_neighber.cpp
  2 | #include <torch/extension.h>
  3 | #include <unordered_set>  // 新增：原文件使用了 unordered_set，但未包含头文件
  4 | #define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
  5 | #define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
  6 | 
  7 | #define AT_DISPATCH_HAS_VALUE(optional_value, ...) \
  8 |     [&] {                                          \
  9 |         if (optional_value.has_value())            \
 10 |         {                                          \
 11 |             const bool HAS_VALUE = true;           \
 12 |             return __VA_ARGS__();                  \
 13 |         }                                          \
 14 |         else                                       \
 15 |         {                                          \
 16 |             const bool HAS_VALUE = false;          \
 17 |             return __VA_ARGS__();                  \
 18 |         }                                          \
 19 |     }()
 20 | 
 21 | torch::Tensor sample_neighber_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
 22 |                int64_t num_neighbors, bool replace);
 23 | 
 24 | // Returns `n_id`
 25 | torch::Tensor sample_neighber_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
 26 |                int64_t num_neighbors, bool replace)
 27 | {
 28 |     CHECK_CPU(rowptr);
 29 |     CHECK_CPU(col);
 30 |     CHECK_CPU(idx);
 31 |     CHECK_INPUT(idx.dim() == 1);
 32 | 
 33 |     auto rowptr_data = rowptr.data_ptr<int64_t>();
 34 |     auto col_data = col.data_ptr<int64_t>();
 35 |     auto idx_data = idx.data_ptr<int64_t>();
 36 | 
 37 |     std::vector<int64_t> n_ids;
 38 | 
 39 |     int64_t n, c, e, row_start, row_end, row_count;
 40 | 
 41 |     if (num_neighbors < 0)
 42 |     {   // No sampling
 43 |         for (int64_t i = 0; i < idx.numel(); i++)
 44 |         {
 45 |             n = idx_data[i];
 46 |             row_start = rowptr_data[n]; row_end = rowptr_data[n + 1];
 47 |             row_count = row_end - row_start;
 48 |             for (int64_t j = 0; j < row_count; j++)
 49 |             {
 50 |                 e = row_start + j;
 51 |                 c = col_data[e];
 52 |                 n_ids.push_back(c);
 53 |             }
 54 |         }
 55 |     }
 56 |     else if (replace)
 57 |     {   // True sampling WITH replacement: 每次独立均匀抽取一个邻居
 58 |         for (int64_t i = 0; i < idx.numel(); i++)
 59 |         {
 60 |             n = idx_data[i];
 61 |             row_start = rowptr_data[n]; row_end = rowptr_data[n + 1];
 62 |             row_count = row_end - row_start;
 63 |             if (row_count <= 0) continue; // 理论上 add_selfloops 后不会发生
 64 |             for (int64_t j = 0; j < num_neighbors; j++)
 65 |             {
 66 |                 e = row_start + (std::rand() % row_count);
 67 |                 c = col_data[e];
 68 |                 n_ids.push_back(c);
 69 |             }
 70 |         }
 71 |     }
 72 |     else
 73 |     {   // Sample WITHOUT replacement via Robert Floyd algorithm
 74 |         for (int64_t i = 0; i < idx.numel(); i++)
 75 |         {
 76 |             n = idx_data[i];
 77 |             row_start = rowptr_data[n]; row_end = rowptr_data[n + 1];
 78 |             row_count = row_end - row_start;
 79 | 
 80 |             std::unordered_set<int64_t> perm;
 81 |             if (row_count <= num_neighbors)
 82 |             {
 83 |                 for (int64_t j = 0; j < row_count; j++) perm.insert(j);
 84 |             }
 85 |             else
 86 |             {
 87 |                 for (int64_t j = row_count - num_neighbors; j < row_count; j++)
 88 |                 {
 89 |                     if (!perm.insert(std::rand() % j).second) perm.insert(j);
 90 |                 }
 91 |             }
 92 |             for (const int64_t &p : perm)
 93 |             {
 94 |                 e = row_start + p;
 95 |                 c = col_data[e];
 96 |                 n_ids.push_back(c);
 97 |             }
 98 |         }
 99 |     }
100 | 
101 |     int64_t N = (int64_t)n_ids.size();
102 |     auto out_n_id = torch::from_blob(n_ids.data(), {N}, col.options()).clone();
103 |     return out_n_id;
104 | }
105 | PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
106 |     m.def("sample_neighber_cpu", &sample_neighber_cpu, "Node neighborhood sampler");
107 | }
```

## File: F:\SomeProjects\SpikeGNNNet2\spikenet\temporal.py

- Extension: .py
- Language: python
- Size: 3147 bytes
- Created: 2025-09-19 03:14:07
- Modified: 2025-09-19 04:33:29

### Code

```python
 1 | import torch
 2 | import torch.nn as nn
 3 | import torch.nn.functional as F
 4 | 
 5 | # --- replace this class in spikenet/temporal.py ---
 6 | 
 7 | import torch
 8 | import torch.nn as nn
 9 | import torch.nn.functional as F
10 | 
11 | class GroupSparseDelay1D(nn.Module):
12 |     """
13 |     组共享 + 稀疏离散延迟核（支持任意 D、groups，不要求整除）
14 |     输入: x [B, T, D] ；输出: y [B, T, D]
15 |     """
16 |     def __init__(self, D: int, T: int, groups: int = 8, delays=(1, 3, 5)):
17 |         super().__init__()
18 |         self.D, self.T = int(D), int(T)
19 |         self.G = max(1, int(groups))
20 |         # 延迟集合
21 |         self.delays = [int(d) for d in delays if int(d) >= 1]
22 |         if len(self.delays) == 0:
23 |             raise ValueError("`delays` must contain at least one positive integer.")
24 | 
25 |         # 每组一套 delay 权重（共享到被分配到该组的所有通道）
26 |         self.weight = nn.Parameter(torch.zeros(self.G, len(self.delays)))
27 | 
28 |         # 通道 -> 组 的映射（不要求均匀整除，采用 round-robin 分配）
29 |         ch2g = torch.arange(self.D, dtype=torch.long) % self.G
30 |         self.register_buffer("ch2g", ch2g)
31 | 
32 |         self.reset_parameters()
33 | 
34 |     def reset_parameters(self):
35 |         nn.init.normal_(self.weight, mean=0.01, std=0.02)
36 | 
37 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
38 |         # x: [B, T, D]
39 |         B, T, D = x.shape
40 |         if T != self.T or D != self.D:
41 |             raise ValueError(f"Input has shape [B,{T},{D}], but module was built with T={self.T}, D={self.D}")
42 | 
43 |         # 计算每通道的延迟权重：Wg[G,K] -> Wc[D,K]
44 |         Wg = self.weight.softmax(dim=-1)        # [G, K]
45 |         Wc = Wg.index_select(0, self.ch2g)      # [D, K]
46 | 
47 |         # 构造按时间维的移位堆栈: xs -> [B, T, D, K]
48 |         xs = []
49 |         for d in self.delays:
50 |             # 在时间维(=1)左侧 pad d，实现 x_{t-d}
51 |             shifted = F.pad(x, (0, 0, d, 0))[:, :T, :]   # [B, T, D]
52 |             xs.append(shifted.unsqueeze(-1))             # [B, T, D, 1]
53 |         Xstk = torch.cat(xs, dim=-1)                     # [B, T, D, K]
54 | 
55 |         # 通道共享权重按组广播并聚合
56 |         y = (Xstk * Wc.view(1, 1, D, -1)).sum(dim=-1)    # [B, T, D]
57 |         return y
58 | 
59 | 
60 | 
61 | class TemporalSeparableReadout(nn.Module):
62 |     """
63 |     可分离时序读出：Depthwise(时间) + Pointwise(1×1) + 全局均值池化 + FC
64 |     输入 [B, T, D] → 输出 [B, C]
65 |     将原 Linear(T*D, C) 的参数/计算解耦到与 T 基本无关。
66 |     """
67 |     def __init__(self, D: int, C: int, k: int = 5):
68 |         super().__init__()
69 |         pad = k // 2
70 |         # 用 [B, D, T] 的通道为 D 的 depthwise 卷积
71 |         self.dw = nn.Conv1d(D, D, kernel_size=k, padding=pad, groups=D, bias=False)
72 |         self.pw = nn.Conv1d(D, D, kernel_size=1, bias=False)
73 |         self.fc = nn.Linear(D, C)
74 | 
75 |     def forward(self, spikes: torch.Tensor) -> torch.Tensor:
76 |         # spikes: [B, T, D]
77 |         x = spikes.transpose(1, 2)        # [B, D, T]
78 |         x = self.pw(self.dw(x))           # [B, D, T]
79 |         x = x.mean(dim=-1)                # [B, D]
80 |         return self.fc(x)
```

## File: F:\SomeProjects\SpikeGNNNet2\spikenet\utils.py

- Extension: .py
- Language: python
- Size: 2759 bytes
- Created: 2025-09-19 01:31:58
- Modified: 2023-09-27 17:42:24

### Code

```python
 1 | import numpy as np
 2 | import scipy.sparse as sp
 3 | import torch
 4 | from sample_neighber import sample_neighber_cpu
 5 | from texttable import Texttable
 6 | 
 7 | try:
 8 |     import torch_cluster
 9 | except ImportError:
10 |     torch_cluster = None
11 | 
12 | 
13 | def set_seed(seed):
14 |     np.random.seed(seed)
15 |     torch.manual_seed(seed)
16 |     torch.cuda.manual_seed(seed)
17 | 
18 | def tab_printer(args):
19 |     """Function to print the logs in a nice tabular format.
20 |     
21 |     Note
22 |     ----
23 |     Package `Texttable` is required.
24 |     Run `pip install Texttable` if was not installed.
25 |     
26 |     Parameters
27 |     ----------
28 |     args: Parameters used for the model.
29 |     """
30 |     args = vars(args)
31 |     keys = sorted(args.keys())
32 |     t = Texttable() 
33 |     t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," "), args[k]] for k in keys])
34 |     print(t.draw())
35 | 
36 |     
37 | class Sampler:
38 |     def __init__(self, adj_matrix: sp.csr_matrix):
39 |         self.rowptr = torch.LongTensor(adj_matrix.indptr)
40 |         self.col = torch.LongTensor(adj_matrix.indices)
41 | 
42 |     def __call__(self, nodes, size, replace=True):
43 |         nbr = sample_neighber_cpu(self.rowptr, self.col, nodes, size, replace)
44 |         return nbr
45 |     
46 |     
47 | class RandomWalkSampler:
48 |     def __init__(self, adj_matrix: sp.csr_matrix, p: float = 1.0, q: float = 1.0):
49 |         self.rowptr = torch.LongTensor(adj_matrix.indptr)
50 |         self.col = torch.LongTensor(adj_matrix.indices)
51 |         self.p = p
52 |         self.q = q
53 |         assert torch_cluster, "Please install 'torch_cluster' first."
54 | 
55 |     def __call__(self, nodes, size, replace=True):
56 |         nbr = torch.ops.torch_cluster.random_walk(self.rowptr, self.col, nodes, size, self.p, self.q)[0][:, 1:] 
57 |         return nbr
58 | 
59 | 
60 | def eliminate_selfloops(adj_matrix):
61 |     """eliminate selfloops for adjacency matrix.
62 | 
63 |     >>>eliminate_selfloops(adj) # return an adjacency matrix without selfloops
64 | 
65 |     Parameters
66 |     ----------
67 |     adj_matrix: Scipy matrix or Numpy array
68 | 
69 |     Returns
70 |     -------
71 |     Single Scipy sparse matrix or Numpy matrix.
72 | 
73 |     """
74 |     if sp.issparse(adj_matrix):
75 |         adj_matrix = adj_matrix - sp.diags(adj_matrix.diagonal(), format='csr')
76 |         adj_matrix.eliminate_zeros()
77 |     else:
78 |         adj_matrix = adj_matrix - np.diag(adj_matrix)
79 |     return adj_matrix
80 | 
81 | 
82 | def add_selfloops(adj_matrix: sp.csr_matrix):
83 |     """add selfloops for adjacency matrix.
84 | 
85 |     >>>add_selfloops(adj) # return an adjacency matrix with selfloops
86 | 
87 |     Parameters
88 |     ----------
89 |     adj_matrix: Scipy matrix or Numpy array
90 | 
91 |     Returns
92 |     -------
93 |     Single sparse matrix or Numpy matrix.
94 | 
95 |     """
96 |     adj_matrix = eliminate_selfloops(adj_matrix)
97 | 
98 |     return adj_matrix + sp.eye(adj_matrix.shape[0], dtype=adj_matrix.dtype, format='csr')
```

