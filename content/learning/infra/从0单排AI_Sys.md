---
title: '从0单排AI_Sys'
date: 2025-02-12T01:15:28+08:00
slug:
summary:
description: 月球大叔的 AI_infra 系列视频
cover:
    image:
    alt:
    caption:
    relative: false
showtoc: true
draft: true
tags: ['infra']
categories:
---

## 0. 介绍

ML Sys -> LLM Sys -> domain specific Sys

(data -> training -> serving) & platform


## 1. Clipper 文章讲解
Serving 文章
Sky Computing Lab in UCB
Ion Stoica

什么是 ML/DL serving
如何优化
学术 vs 工业

training -> serving
优化指标
1. latency 
   1. P99
   2. P50
2. Throughput
   1. qps 


## 2. clipper　对吞吐量优化
latency & througput & accuracy

串行处理快，但是浪费资源，并行被浪费
batching：批处理队列
问题：延迟 = 打包 + 处理，打包时间太长

-> adaptive batching
1. dynamic batchsize:
batch不固定，根据负载变化
AIMD: 加性增加，惩罚减少
SLO （latency）
变化太快？-> 

2. delayed batching
   batch 没填满等 2ms，填满了再发送

动态调整batchsize


## 3. clipper 对延迟的优化
计算消耗时间
LRU -> clock 算法

LRU cache algorithm


## 4. clipper 用系统方法提高模型acc

model selection 
花费和准确率 trade-off
躲避老虎机问题

model <- weight
input -> weight -> model -> loss -> update weight(EXP3)

ensemble multiple model results
Q: loss? AB testing / sampling / mannual retest


Clipper vs tensorflow-serving
- features
- usibility (interface, vehichlization?)


## 5. 4 key component of AI platform

Studio: jupyter
Data: 1. data source 2. data processing 3. fature store 4. data labling
Training: 1. training task management 2. AutoML 3. experiments management
Deployment 1. model registration 2. model deployment 3. Online monitoring


## 6/7. Works of AI Platform Engineering
computation & networking & storage

1. computation:
   1. PC
   2. Cluster
      1. scheduling
      2. sharing
      3. scaling
   3. metrics:
      1. JCT
      2. GPU Hours
      3. Makespan
   4. for LLMs
      1. data paral
      2. pipeline
      3. tensor
      4. hybird
2. networking
   1. collective communication
      1. all reduce
      2. all together
   2. communication scheduling
      1. bytescheduler
      2. compression
      3. monitoring, profile
3. storage
   1. stage 1: load training data
      1. scalable
      2. abstract/unified interface
      3. cache 
   2. stage 2: save ckpoints
      1. fluid, alluxio
      2. scalable
      3. broadcast skpt partition


## 8. Falure analysis
1. infra / hardware issues
   1. temperature
   2. infiniband slow network
   3. falure on disks
2. software
   1. version
   2. networking bt gp
3. coding
   1. import package
   2. invalid admission
   3. device
   4. modality format
4. numerical, parameter issue
   1. batchsize
   2. learning rate


## 容错系统
Failure analysis
Exception Monitoring
- monitor & history data
- lightweighted detection  & self-diagnoisis

Recovery
- checkpointing
- hot transfer
- backups