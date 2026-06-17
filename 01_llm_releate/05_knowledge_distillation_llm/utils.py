import torch
"""
数据采样区别： 正向 KL 是用老师数据教学生（被动学, 监督微调或者离线数据蒸馏（黑盒蒸馏就是典型的）），反向 KL 是学生自己生成数据，让老师来评判（主动学）。
采样者： 正向 KL 主要基于教师模型采样；反向 KL 基于学生模型采样
# https://zhuanlan.zhihu.com/p/1996277720846471471
"""
# 计算前向kl散度
def compute_fkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id,
        reduction="sum",
        temp = 1.0, 
        
    ):
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (teacher_probs * (teacher_log_probs - log_probs)) 
        kl = kl.sum(-1)
        if reduction == "sum":
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            kl = kl.sum()

        return kl
# 计算反向kl散度
def compute_rkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id,
        reduction="sum", 
        temp = 1.0
    ):
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (probs * (log_probs - teacher_log_probs))
        kl = kl.sum(-1)
        if reduction == "sum":
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            kl = kl.sum()
        return kl

# 计算偏向前kl散度
def compute_skewed_fkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id, 
        reduction="sum", 
        temp = 1.0,
        skew_lambda = 0.1
    ):
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = skew_lambda * teacher_probs + (1 - skew_lambda) * probs
        mixed_log_probs = torch.log(mixed_probs)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (teacher_probs * (teacher_log_probs - mixed_log_probs))
        kl = kl.sum(-1)
        if reduction == "sum":
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            kl = kl.sum()

            
        return kl
# 计算偏向反kl散度    
def compute_skewed_rkl(
    logits, 
    teacher_logits, 
    target,
    padding_id,
    reduction="sum", 
    temp = 1.0,
    skew_lambda = 0.1
):
    logits = logits / temp
    teacher_logits = teacher_logits / temp
    
    probs = torch.softmax(logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    mixed_probs = (1 - skew_lambda) * teacher_probs + skew_lambda * probs
    mixed_log_probs = torch.log(mixed_probs)
    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
    kl = (probs * (log_probs - mixed_log_probs))
    kl = kl.sum(-1)
    
    if reduction == "sum":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        kl = kl.sum()


    return kl