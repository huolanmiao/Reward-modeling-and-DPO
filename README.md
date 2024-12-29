# 奖励模型

## 部署align-anything，训练奖励模型

### 偏好数据集键值转换

- Reward model接受`prompt+answer`作为输入，返回得分
- 模仿`PKUSafeRLHF`模板，完成`HOMEWORK`模板
- 在`RMTrainer`中，键值转换发生在定义`PreferenceDataset`的时候

### 训练Reward Model

- `reward model`是一个添加`reward_head`的模型`T2TRewardModel`，我们考虑`attention mask`，采用最后一个token的得分作为最终得分
- 训练时我们输入`better_input_ids`和`worse_input_ids`，然后用`reward model`分别计算其得分
- 优化如下loss function：`loss = -F.logsigmoid(higher_end_reward - lower_end_reward).mean()`

```python
# Relevant codes
self.score_head = nn.Linear(config.hidden_size, 1, bias=False)
last_hidden_state = outputs.hidden_states[-1]
scores = self.score_head(last_hidden_state).float()
```
<figure style="text-align: center;">
  <img src="./fig/rm_loss.bmp" alt="RM loss" />
  <figcaption>RM loss</figcaption>
</figure>

- `tensorboard --logdir ./output/rm`可以看到训练曲线
<div style="display: flex; justify-content: space-around;">
  <figure style="text-align: center;">
    <img src="./fig/reward_model_train_acc.bmp" alt="图片1" width="400" />
    <figcaption>RM training accuracy</figcaption>
  </figure>
  
  <figure style="text-align: center;">
    <img src="./fig/reward_model_train_loss.bmp" alt="图片2" width="400" />
    <figcaption>RM training loss</figcaption>
  </figure>

  <figure style="text-align: center;">
    <img src="./fig/reward_model_eval_acc.bmp" alt="图片3" width="400" />
    <figcaption>RM eval accuracy</figcaption>
  </figure>
</div>

### 评测奖励模型

#### 运行结果

- 将训练集置空可以直接在向`deepspeed`传参的时候删去`--train_datasets`
- 把模型路径设置为`../output/rm/slice_end`
- 从测试集的示例可以看出，我们的奖励模型为更加helpful的回答，赋予更高的reward 
<figure style="text-align: center;">
  <img src="./fig/eval.bmp" alt="eval" />
  <figcaption>Evaluation</figcaption>
</figure>
- 从训练是的评估曲线和最后的评估结果可以看出，没有出现过拟合现象。但是在测试集上准确率低于训练集，这说明模型泛化能力不足，这可能是参数量太小导致的

#### 踩坑记录

我将训练集置空`TRAIN_DATASETS= ""`，把模型路径设置为`../output/rm/slice_end`，然后运行`bash rm.sh`，出现如下报错：
```
[rank0]:   File "/home/pku0016/hw2/align-anything/align_anything/utils/template_registry.py", line 30, in get_template_class
[rank0]:     raise ValueError(f"Template '{template_name}' not found.")
[rank0]: ValueError: Template 'None' not found.
```

#### `TEMPLATE`是怎么传入`PreferenceDataset`？
```python
# 在基类SupervisedTrainerBase的get_dataloaders()，会设置PreferenceDataset的template等参数
def init_datasets(self) -> None:
    """Initialize training and evaluation datasets."""
    self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
        PreferenceDataset, PreferenceDataset
    )
```

#### 分析`SupervisedTrainerBase.get_dataloaders()`
- 接受两个数据集的类型，然后根据`cfgs.data_cfgs`中的`路径`和`template`等参数设置数据集
- 先前的问题在于，我们置空训练集期望的是不进入`if self.cfgs.data_cfgs.train_datasets:`，但是报错显示在其中的`get_template_class`函数。只说明我们置空的方式有问题，应该将其设置为`None`而不是`""`

```python
def get_dataloaders(self, train_data_dtype, eval_data_dtype) -> None:
  """Get the dataloaders based on data_dtype."""
  train_dataloader = []
  eval_dataloader = []
  if self.cfgs.data_cfgs.train_datasets:
      self.train_template = get_template_class(self.cfgs.data_cfgs.train_template)
      train_dataset = train_data_dtype(
          path=self.cfgs.data_cfgs.train_datasets,
          template=self.cfgs.data_cfgs.train_template,
          ···
      )
      train_dataloader = DataLoader(
          train_dataset,
          ···
      )
  if self.cfgs.data_cfgs.eval_datasets:
      self.eval_template = get_template_class(self.cfgs.data_cfgs.eval_template)
      eval_dataset = eval_data_dtype(
          path=self.cfgs.data_cfgs.eval_datasets,
          template=self.cfgs.data_cfgs.eval_template,
          ···
      )
      eval_dataloader = DataLoader(
          eval_dataset,
          ···
      )

  return train_dataloader, eval_dataloader
```

### 可视化偏好数据集

- `RMScore.eval()`中使用`dataloader`加载`SupervisedDataset`，其中`text`是prompt与response的拼接，与`PreferenceDataset`中的形式类似
- 将`text`输入`Reward Model`进行评分，然后将结果与`prompt`和`response`一并存入`.json`文件
  ```python
  class RMScore(SupervisedTrainerBase):
    @torch.no_grad()
    def eval(self) -> dict[str, Any]:
        ··· 
        rewards = []
        prompts = []
        responses = []
        batch = None
        for batch in eval_dataloader:
            # Calculate the score
            output = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            end_scores = output.end_scores
            rewards.append(end_scores)
            # Get the prompt and response
            decoded_prompt_and_response = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            prompt, response = split_prompt_response(decoded_prompt_and_response, split_token=self.eval_template.split_token)
            prompts.append(prompt)
            responses.append(response)
        ···
  ```
- 更改template，运行`bash rm_score.sh`。最终得到两个`.json`文件，分别是人类标注的好回复和坏回复的测试结果
- 可视化`Score`的分布，可以看到未经过奖励模型训练的`qwen`并不能分辨人类的偏好，而训练好的`Reward Model`捕捉到人类偏好，为人类选择的回复赋予更高的分数
<figure style="text-align: center;">
  <img src="./fig/score_distribution_qwen.png" alt="score_distribution_qwen" />
  <figcaption>Score distribution of human preference data by qwen</figcaption>
</figure>


<figure style="text-align: center;">
  <img src="./fig/score_distribution_rm.png" alt="score_distribution_rm" />
  <figcaption>Score distribution of human preference data by trained reward model</figcaption>
</figure>

- 如果在训练集上，用奖励模型评分，则人类标注为`chosen`和`rejected`的数据的评分分布差距更高
<figure style="text-align: center;">
  <img src="./fig/trainset_rm.png" alt="train" />
  <figcaption>Score distribution on training set</figcaption>
</figure>

## 回答问题

### 奖励建模有哪些应用？

在RLHF中，我们根据人类偏好训练奖励模型，然后用奖励模型作为RL的反馈信号，即对模型的输出进行评分，作为奖励信号的一部分，finetune the supervised policy.
<figure style="text-align: center;">
  <img src="./fig/rlhf.bmp" alt="RLHF pipeline" />
  <figcaption>RLHF pipeline</figcaption>
</figure>


### 奖励建模训练可能存在哪些鲁棒性问题？
- 我们根据人类标注的偏好对训练奖励模型，如果输入奖励模型的`prompt+answer`是OOD的，那么得到的评分很可能并不准确，因为奖励模型很可能不能在新的空间中捕捉人类偏好。
- repeating data 会导致过拟合。RLHF中对同一个输入有K个回答的标注，进而形成$\binom{K}{2}$个偏好对。如果将这些偏好对视作独立的数据点，shuffle之后进行训练，那么会导致每一种回答都进行了K-1次梯度优化，进而导致过拟合，所以应该作为一个batch放到一个forward pass中。 [Ouyang, Long, et al.](https://arxiv.org/pdf/2203.02155)
  
<figure style="text-align: center;">
  <img src="./fig/rlhf_rm.bmp" alt="RLHF rm" />
  <figcaption>Loss function for reward modeling</figcaption>
</figure>

### 如何缓解奖励建模的长度偏差？


### 有哪些拟合多元人类偏好的奖励建模方法？

# DPO微调
- `tensorboard --logdir ./output/dpo`可以看到训练曲线

<div style="display: flex; justify-content: space-around;">
  <figure style="text-align: center;">
    <img src="./fig/dpo_acc.bmp" alt="dpo_acc" width="400" />
    <figcaption>DPO train accuracy</figcaption>
  </figure>
  
  <figure style="text-align: center;">
    <img src="./fig/dpo_train_loss.bmp" alt="dpo_train_loss" width="400" />
    <figcaption>DPO train loss</figcaption>
  </figure>
</div>

- reward accuracy反映出当前模型生成人类选择文本的log_prob大于未被选择文本的概率
```python
# better_log_ratio是当前模型与参考模型对当前文本建模的log_prob之差
better_log_ratio = better_log_prob - ref_better_log_prob
worse_log_ratio = worse_log_prob - ref_worse_log_prob
# DPO loss function
losses.append(
    -F.logsigmoid(
        self.cfgs.train_cfgs.scale_coeff * (better_log_ratio - worse_log_ratio),
    ),
)
# reward是better_log_ratio乘以一个scale_coeff得到的
better_sample_rewards.append(
    self.cfgs.train_cfgs.scale_coeff * better_log_ratio.detach(),
)
···
reward_accuracy = (better_sample_reward > worse_sample_reward).float().mean() 
```
## DPO model inference

### Respond to the test prompts
- 分别使用DPO训练好的模型和qwen在用于**测试**的prompts上生成回复，然后用`Reward Model`打分，绘制得分的分布
<figure style="text-align: center;">
  <img src="./fig/qwen_dpo_4_test_response.png" alt="test_response" />
  <figcaption>Use the reward model to grade the response to testing prompts</figcaption>
</figure>

### Respond to the randomly sampled training prompts
- 分别使用DPO训练好的模型和qwen在**训练集**中随机选取的prompts上生成回复，然后用`Reward Model`打分，绘制得分的分布
<figure style="text-align: center;">
  <img src="./fig/qwen_dpo_4_train_response.png" alt="train_response" />
  <figcaption>Use the reward model to grade the response to training prompts</figcaption>
</figure>

## Analysis
- 我们发现dpo训练后的模型，无论在测试集还是随机选取的训练集中的prompt上，`reward model`打分的分布都小于原始的qwen模型
- 我认为原因在于**奖励模型不准确**
  - 从DPO训练曲线可以看出，DPO更倾向于生成人类选择的文本，所有训练是有效的
  - 先前测试reward model的时候，分数的分布集中于0附近，而此时的评分集中在10附近。我认为这说明reward model对新生成的回复的打分是不准确的，不能泛化到OOD的数据
# 参考文献

@article{ouyang2022training,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeffrey and Jiang, Xu and Almeida, Diogo and Wainwright, Carroll and Mishkin, Pamela and Zhang, Chong and Agarwal, Sandhini and Slama, Katarina and Ray, Alex and others},
  journal={Advances in neural information processing systems},
  volume={35},
  pages={27730--27744},
  year={2022}
}