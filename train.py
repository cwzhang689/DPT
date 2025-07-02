import torch
from torch import nn
from src import model as mm
from src.utils import *
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.eval_metrics import *
import matplotlib.pyplot as plt

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    if hyp_params.pretrained_model is not None:
        model = getattr(mm, "PromptModel")(hyp_params)
        model = transfer_model(model, hyp_params.pretrained_model, hyp_params.use_cuda)
    else:
        model = getattr(mm, "MULTModel")(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=hyp_params.when, factor=0.1, verbose=True
    )
    settings = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
    }
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings["model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    scheduler = settings["scheduler"]

    # 初始化空的列表来记录训练和验证损失
    train_losses = []
    val_losses = []

        # 记录最好的验证损失和对应的 epoch
    best_valid_loss = float('inf')  # 确保初始化为一个很大的数
    best_epoch = -1  # 记录最佳 epoch

    def train(model, optimizer, criterion):
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()

        # 定义梯度累积步骤数
        accumulation_steps = 4  # 可以根据需要调整这个值

        # 将模型迁移到GPU（如果使用GPU）
        model = model.to(model.device)
        model.zero_grad()
        for i_batch, (batch_X, batch_Y, missing_mod) in enumerate(train_loader):
            text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)

            # 将数据迁移到GPU（如果使用GPU）
            if hyp_params.use_cuda:
                text, audio, vision, eval_attr = (
                    text.to(model.device),
                    audio.to(model.device),
                    vision.to(model.device),
                    eval_attr.to(model.device),
                )

            batch_size = text.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            preds = net(text, audio, vision, missing_mod)

            raw_loss = criterion(preds, eval_attr)
            raw_loss.backward()

            # 梯度累积
            if (i_batch + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
                optimizer.step()
                optimizer.zero_grad()

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size

            # 每隔一定批次打印训练进度
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch} | Batch {i_batch}/{num_batches} | Loss {avg_loss}")
                proc_loss, proc_size = 0, 0
                start_time = time.time()
        if proc_size % (accumulation_steps * hyp_params.batch_size) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            optimizer.zero_grad()


    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, missing_mod) in enumerate(loader):
                text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1

                if hyp_params.use_cuda:
                    text, audio, vision, eval_attr = (
                        text.cuda(),
                        audio.cuda(),
                        vision.cuda(),
                        eval_attr.cuda(),
                    )
                    if hyp_params.dataset == "iemocap":
                        eval_attr = eval_attr.long()

                batch_size = text.size(0)
                preds = model(text, audio, vision, missing_mod)

                if hyp_params.dataset == "iemocap":
                    preds = preds.view(-1, 4)
                    eval_attr = eval_attr.view(-1)
                total_loss += criterion(preds, eval_attr).item() * batch_size

                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    # 初始化画图
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")

    # 创建两条曲线：训练和验证损失
    train_line, = ax.plot([], [], label="Train Loss")
    val_line, = ax.plot([], [], label="Validation Loss")
    ax.legend()

    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train(model, optimizer, criterion)
        torch.cuda.empty_cache()  # 每个epoch后清除缓存
        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, _, _ = evaluate(model, criterion, test=True)

        # 记录训练和验证损失
        train_losses.append(val_loss)  # 使用验证损失作为训练损失的代表
        val_losses.append(test_loss)

        # 如果当前验证损失小于最好的验证损失，更新最佳验证损失和对应的 epoch
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_epoch = epoch
            torch.save(model, hyp_params.name)  # 添加保存

        # 动态更新图像
        train_line.set_xdata(range(1, epoch + 1))
        train_line.set_ydata(train_losses)
        val_line.set_xdata(range(1, epoch + 1))
        val_line.set_ydata(val_losses)

        ax.relim()  # 重新计算数据范围
        ax.autoscale_view(True, True, True)  # 自动缩放视图

        plt.draw()  # 绘制更新后的图形
        plt.pause(0.1)  # 暂停0.1秒，显示图像

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)

        print("-" * 50)
        print(f"Epoch {epoch} | Time {duration:.4f} sec | Valid Loss {val_loss:.4f} | Test Loss {test_loss:.4f}")
        print("-" * 50)

    print(f"Best validation loss occurred at epoch {best_epoch} with loss: {best_valid_loss:.4f}")

    model = torch.load(hyp_params.name)
    _, results, truths = evaluate(model, criterion, test=False)

    if hyp_params.dataset == "mosei":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == "mosi":
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == "iemocap":
        eval_iemocap(results, truths)
    elif hyp_params.dataset == "sims":
        eval_sims(results, truths)

    # 关闭交互模式并保存图像
    plt.ioff()  # 关闭交互模式
    plt.savefig('dynamic_loss_curve.png')  # 保存图像到文件
    plt.show()  # 显示最终图像
