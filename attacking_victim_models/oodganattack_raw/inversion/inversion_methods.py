import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from scipy.optimize import minimize
from utils.image_precossing import _tanh_to_sigmoid


def get_inversion(optimization_type, args):
    if optimization_type == 'sgd':
        return GradientDescent(args.iterations, args.lr, optimizer=optim.SGD, args=args)
    elif optimization_type == 'adam':
        return GradientDescent(args.iterations, args.lr, optimizer=optim.Adam, args=args)
    elif optimization_type == 'powell':
        return BBPowell(args.iterations, args=args)

    
# =============================================================================
# GradientDescent
# =============================================================================
class GradientDescent(object):
    def __init__(self, iterations, lr, optimizer, args):
        self.iterations = iterations
        self.lr = lr
        self.optimizer = optimizer
        self.init_type = args.init_type  
        self.args = args

    def invert(self, generator, target_model, ref_labs, ref_preds,
               loss_function, batch_size=1, video=True, *init):
        input_size_list = generator.input_size()
        if len(init) == 0:
            if generator.init is False:  
                latent_estimate = []
                for input_size in input_size_list:
                    if self.init_type == 'Zero':
                        latent_estimate.append(torch.zeros((batch_size,) + input_size).cuda())
                    elif self.init_type == 'Normal':
                        latent_estimate.append(torch.randn((batch_size,) + input_size).cuda())
            else:
                latent_estimate = list(generator.init_value(batch_size))
        else:
            assert len(init) == len(input_size_list), 'Please check the number of init value'
            latent_estimate = init
        for latent in latent_estimate:
            latent.requires_grad = True
        
        optimizer = self.optimizer(latent_estimate, lr=self.lr)
        target_model.eval()
        conf_probs = nn.Softmax(dim=1)
        
        for rept in range(10):       
            
            for i in tqdm(range(self.iterations)):
                gen_samples = generator(latent_estimate)   
                gen_samples = _tanh_to_sigmoid(torch.clamp(gen_samples, -1, 1))
                y_preds =  target_model(gen_samples) 
                
                # early stopping
                if self.args.early_stopping == 'True':
                    if self.args.req_conf == 0:
                        if torch.equal(torch.argmax(y_preds, dim=1), ref_labs) ==True:
                            break
                    else:
                        if torch.equal(torch.argmax(y_preds, dim=1), ref_labs) ==True and torch.max(conf_probs(y_preds), dim=1)[0] >= self.args.req_conf:
                            break 
                
                optimizer.zero_grad()
                if self.args.loss_type == 'lc': 
                    loss = loss_function(y_preds, ref_labs)     
                else:
                    loss = loss_function(y_preds, ref_preds)     
                loss.backward()
                optimizer.step()
                   
            if i < self.iterations -1:
                return latent_estimate
            else:
                print('restart times', rept+1)
                latent_estimate.append(torch.randn((batch_size,) + input_size).cuda())  # restart
                for latent in latent_estimate:
                    latent.requires_grad = True
                optimizer = self.optimizer(latent_estimate, lr=self.lr*0.5)                
                            
        return latent_estimate

# =============================================================================
# BBPowell
# =============================================================================
class BBPowell(object):
    def __init__(self, iterations, args):
        self.iterations = iterations
        self.init_type = args.init_type  # ['Zero', 'Normal']
        self.args = args
        self.query_cnt = 0   

    def invert(self, generator, target_model, ref_labs, ref_preds,
               loss_function, batch_size=1, video=True, *init):
        
        input_size_list = generator.input_size()
        if len(init) == 0:
            if generator.init is False:  # initilize
                latent_estimate = []
                for input_size in input_size_list:
                    if self.init_type == 'Zero':
                        latent_estimate = np.zeros(batch_size,512)
                    elif self.init_type == 'Normal':
                        latent_estimate = np.random.randn(batch_size,512)
                     
            else:
                latent_estimate = list(generator.init_value(batch_size))
        else:
            assert len(init) == len(input_size_list), 'Please check the number of init value'
            latent_estimate = init

        target_model.eval()
        conf_probs = nn.Softmax(dim=1)

        
        # if an initial value can succeed, do not need to optimize
        gen_samples = generator([torch.from_numpy(latent_estimate).float().view(-1, 512).cuda()])           
        gen_samples = _tanh_to_sigmoid(torch.clamp(gen_samples, -1, 1)) 
        y_preds =  target_model(gen_samples) 
        self.query_cnt = 1   
        
        if self.args.early_stopping == 'True':
            if self.args.req_conf == 0:
                if torch.equal(torch.argmax(y_preds, dim=1), ref_labs) ==True:
                    return latent_estimate
            else:
                if torch.equal(torch.argmax(y_preds, dim=1), ref_labs) ==True and torch.max(conf_probs(y_preds), dim=1)[0] >= self.args.req_conf:
                    return latent_estimate
        
        options = {'maxiter': self.args.maxiter,
           'maxfev': self.args.maxfev,
           'return_all': True,
           'disp': True,
           'xtol': self.args.xtol,
           'ftol': self.args.ftol}

        
        def objective(z):
            z_ = [torch.from_numpy(z).float().view(-1, 512).cuda()]
            self.query_cnt += 1
            y_ =  generator(z_)
            y_ = _tanh_to_sigmoid(torch.clamp(y_, -1, 1)) 
            y_preds =  target_model(y_)  

            if self.args.early_stopping == 'True':
                if self.args.req_conf == 0:
                    if torch.equal(torch.argmax(y_preds, dim=1), ref_labs) ==True:
                        return 0
                else:
                    if torch.equal(torch.argmax(y_preds, dim=1), ref_labs) ==True and torch.max(conf_probs(y_preds), dim=1)[0] >= self.args.req_conf:
                        return 0

            if self.args.loss_type == 'lc': 
                loss = loss_function(y_preds, ref_labs)     
            else:
                loss = loss_function(y_preds, ref_preds)                 
            final_loss = torch.mean(loss)
            final_loss = final_loss.detach().cpu().numpy()   
            return final_loss   
        

        res = minimize(objective, latent_estimate, method='Powell', options=options)
        latent_estimate = res.x
        print('res.nfev', res.nfev)
        
        
        return latent_estimate

