import os
import csv

def prepare_csv(trainlog_path, testlog_path, minimizer, rho):
    if not os.path.exists(trainlog_path):
        os.makedirs(trainlog_path)
    if not os.path.exists(testlog_path):
        os.makedirs(testlog_path)
    csv_train = open(trainlog_path+'/'+minimizer.lower()+str(rho)+'.csv', 'a+', newline='')
    csv_train_writer = csv.writer(csv_train)
    csv_test = open(testlog_path+'/'+minimizer.lower()+str(rho)+'.csv', 'a+', newline='')
    csv_test_writer = csv.writer(csv_test)
    return csv_train, csv_train_writer, csv_test, csv_test_writer

def write_csv(csv_train,csv_train_writer,csv_test,csv_test_writer,head=False,train=False,test=False,args=None,**kwargs):
    if head:
        csv_train_writer.writerow([
            'network:',args.network,'depth',args.depth,'Loss:CrossEntropy','DataSet:',args.dataset,'LearningRate:',
            args.learning_rate,'BatchSize:',args.train_batch_size,'EpochRange:',args.epoch_range,'Optimizer:SGD','Minimizer:',args.minimizer])
        csv_train_writer.writerow(['Epoch','Train_Loss', 'Disturb_Loss', 'Train_Accuracy'])
        csv_train.flush()
        csv_test_writer.writerow([
            'network:',args.network,'depth',args.depth,'Loss:CrossEntropy','DataSet:',args.dataset,'Optimizer:SGD','Minimizer:',args.minimizer])
        csv_test_writer.writerow(['Epoch', 'Test_Loss','Test_Accuracy','Generalization_Gap'])
        csv_test.flush()
    if train:
        csv_train_writer.writerow([kwargs['epoch']+1, 
                                   kwargs['train_loss'] / (kwargs['batch_index'] + 1), 
                                   kwargs['disturb_loss'] / (kwargs['batch_index'] + 1), 
                                   100.*kwargs['num_correct'] / len(kwargs['trainloader'].dataset)])
        csv_train.flush()
    if test:
        csv_test_writer.writerow([kwargs['epoch']+1,
                                  kwargs['test_loss'] / (kwargs['batch_index']+1), 
                                  100.*kwargs['num_correct'] / len(kwargs['testloader'].dataset), 
                                  abs((kwargs['test_loss'] / (kwargs['batch_index']+1)) - (kwargs['train_loss']))])
        csv_test.flush()