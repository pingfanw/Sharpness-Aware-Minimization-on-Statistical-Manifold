
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.loss import LabelSmoothingCrossEntropy
from utils.data_utils import NoiseDataLoader
from utils.smsam_utils import get_param_norm, get_fisher_trace
from optimizer import sgd_mod
from optimizer.minimizer import SMSAM, ASAM, SAM
from utils.get_networks import get_network
from utils.log_utils import prepare_csv, write_csv


# ----------train----------#
def train(args):

    noisedataloader = NoiseDataLoader(args.dataset.lower(), 
                                      args.noise_rate, 
                                      args.noise_mode, 
                                      args.train_batch_size, 
                                      args.test_batch_size, 
                                      args.num_workers, 
                                      args.data_path)
    trainloader, testloader = noisedataloader.get_loader()
    net = get_network(args)
    if args.smoothing:
        loss_function = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        loss_function = nn.CrossEntropyLoss()
    optimizer = sgd_mod.SGD(net, args.minimizer, 
                                lr=args.learning_rate, 
                                weight_decay=args.weight_decay, 
                                momentum=args.momentum, 
                                stat_decay=args.stat_decay, 
                                damping=args.damping, 
                                batch_averaged=args.batch_averaged, 
                                TCov=args.TCov, TInv=args.TInv)

    minimizer = eval(args.minimizer)(optimizer, net, rho=args.rho, eta=args.eta)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, args.epoch_range)

    csv_train, csv_train_writer, csv_test, csv_test_writer = prepare_csv(args.trainlog_path, args.testlog_path, args.minimizer, args.rho)
    write_csv(csv_train,csv_train_writer,csv_test,csv_test_writer,head=True,train=False,test=False,args=args)

    # train
    optimizer.acc_stats = True
    for epoch in range(args.epoch_range):
        total = 0
        num_correct = 0
        train_loss = 0.0
        disturb_loss = 0.0
        net.train()

        desc = ('[Train][%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' % 
                (args.minimizer.lower(), scheduler.get_last_lr(), 0, 0, num_correct, total))
        prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, position=0, leave=True)
        for batch_index, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            if args.network == 'mlp': inputs = inputs.view(-1, 784)  # flatten the inputs for MLP
            outputs = net.forward(inputs)
            predictions = net(inputs)
            if args.minimizer == "SMSAM":
                # Ascent Step
                loss = loss_function(outputs, labels)
                optimizer.acc_stats = True
                optimizer.zero_grad()
                loss.backward()
                # compute regularized term before adding the disturb
                param_norm = get_param_norm(net,args.device)
                fisher_trace = get_fisher_trace(net,args.train_batch_size,args.device)
                minimizer.ascent_step()
                # Descent Step
                loss_eps = loss_function(net(inputs), labels)+(args.weight_decay/2 * param_norm * fisher_trace)
                loss_eps.backward()
                optimizer.acc_stats = False
                minimizer.descent_step()
            else:
                # Ascent Step
                loss = loss_function(outputs, labels)
                loss.backward()
                minimizer.ascent_step()
                # Descent Step
                loss_eps = loss_function(net(inputs), labels)
                loss_eps.backward()
                minimizer.descent_step()
            # visualizing and saving stuff
            predictions = outputs.argmax(dim=1) 
            total += labels.size(0)
            num_correct += torch.eq(predictions,labels).sum().item()  
            train_loss += (loss.item())
            disturb_loss += (loss_eps.item())
            desc = ('[Train][%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (args.minimizer.lower(), 
                     scheduler.get_last_lr(), 
                     train_loss / (batch_index + 1), 
                     100. * num_correct / total, 
                     num_correct, total))
            prog_bar.set_description(desc, refresh=True)
            prog_bar.update()
        prog_bar.close()
        write_csv(csv_train,csv_train_writer,csv_test,csv_test_writer,train=True,
                  epoch=epoch,train_loss=train_loss,disturb_loss=disturb_loss,
                  num_correct=num_correct,trainloader=trainloader,batch_index=batch_index)
        scheduler.step()

        # test  
        net.eval()
        total = 0
        num_correct = 0
        test_loss = 0.0
        with torch.no_grad():
            desc = ('[Test][%s][LR=%s] Loss: %.3f | Gap: %.3f | Acc: %.3f%% (%d/%d)' % 
                    (args.minimizer.lower(), scheduler.get_last_lr(), test_loss/(0+1), 0, 0, num_correct, total))
            prog_bar_test = tqdm(enumerate(testloader), total=len(testloader), desc=desc, position=0, leave=True)
            for batch_index, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                if args.network == 'mlp': inputs = inputs.view(-1, 784)  # flatten the inputs for MLP
                outputs = net.forward(inputs)
                predictions = outputs.argmax(dim=1)
                total += labels.size(0)
                num_correct += torch.eq(predictions, labels).sum().item()
                test_loss += loss_function(outputs, labels).item()
                desc = ('[Test][%s][LR=%s] Loss: %.3f | Gap: %.3f | Acc: %.3f%% (%d/%d)' %
                        (args.minimizer.lower(), 
                         scheduler.get_last_lr(), 
                         test_loss / (batch_index + 1), 
                         abs((test_loss / (batch_index+1)) - (train_loss)), 
                         100. * num_correct / total, 
                         num_correct, total))
                prog_bar_test.set_description(desc, refresh=True)
                prog_bar_test.update()
        prog_bar_test.close()
        write_csv(csv_train,csv_train_writer,csv_test,csv_test_writer,test=True,
                  epoch=epoch,test_loss=test_loss,num_correct=num_correct,
                  trainloader=trainloader,batch_index=batch_index,testloader=testloader,train_loss=train_loss)
    csv_train.close()
    csv_test.close()


