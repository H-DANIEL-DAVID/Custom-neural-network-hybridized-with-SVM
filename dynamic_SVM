print("--- Dynamic SVM ---")

class dynamic_SVM:
    def __init__(self, inp, hid, out, l):
        self.w_lst = []
        self.b_lst = []
        self.v_lst = []
        self.bv_lst = []
        self.wf_lst = []
        self.bf_lst = []
        self.h_lst = []
        self.z_lst = []
        self.de_lst = []
        self.loss = []
       
        for i in range(l + 1):
            if i == 0:
                w = np.random.randn(inp, hid) * np.sqrt(2 / inp)
                b = np.zeros((1, hid))
            elif i == l:
                w = np.random.randn(hid, out) * np.sqrt(2 / hid)
                b = np.zeros((1, out))
            else:
                w = np.random.randn(hid, hid) * np.sqrt(2 / hid)
                b = np.zeros((1, hid))
             
            v = np.zeros_like(w)
            bv = np.zeros_like(b)
            
            self.w_lst.append(w)
            self.b_lst.append(b)
            self.v_lst.append(v)
            self.bv_lst.append(bv)
            self.wf_lst.append(v.copy())
            self.bf_lst.append(bv.copy())

    def relu(self, x):
        return np.where(x > 0, x, 0.01 * x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1.0, 0.01)

    def mask(self, x):
        return np.where(x > 0, 1.0, 0.0)

    def train(self,x,y,batch_size,lr = 0.001,g = 0.9,epochs = 10000,c = 100):
        N = x.shape[0]
        for i in range(1,epochs+1):
            indices = np.random.permutation(N)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            for q in range(0,N,batch_size):
             X = x_shuffled[q:q+batch_size]
             Y = y_shuffled[q:q+batch_size]
             mini = X.shape[0]
             for n in range(len(self.w_lst)):
              self.wf_lst[n] = self.w_lst[n] - self.v_lst[n]*g
              self.bf_lst[n] = self.b_lst[n] - self.bv_lst[n]*g
             h = X@self.wf_lst[0] + self.bf_lst[0]
             self.h_lst.append(h)
             z = self.relu(h)
             self.z_lst.append(z)
             for j in range(1,len(self.w_lst)):
                h = z@self.wf_lst[j] + self.bf_lst[j]
                self.h_lst.append(h)
                if j == len(self.w_lst)-1:
                    margin = Y*h
                    y_pred = 1 - margin
                else:
                    z = self.relu(h)
                    self.z_lst.append(z)
             e = self.mask(y_pred)
             de = e*(-Y)
             self.de_lst.append(de)
             for k in range(1,len(self.w_lst)):
                e = de@self.wf_lst[-k].T
                de = e*self.relu_derivative(self.h_lst[-k-1])
                self.de_lst.append(de)
             self.de_lst = self.de_lst[::-1]
             grad = ((X.T@self.de_lst[0]) + (1/c)*self.w_lst[0])/mini
             self.v_lst[0] = (grad*lr + self.v_lst[0]*g)
             self.w_lst[0] -= self.v_lst[0]
             
             
             grad = np.sum(self.de_lst[0], axis = 0,keepdims = True)
             self.bv_lst[0] = grad*lr + self.bv_lst[0]*g
             self.b_lst[0] -= self.bv_lst[0]
             for m in range(1,len(self.w_lst)):
                grad = ((self.z_lst[m-1].T@self.de_lst[m]) + (1/c)*self.w_lst[m])/mini
                self.v_lst[m] = grad*lr + self.v_lst[m]*g
                self.w_lst[m] -= self.v_lst[m]
                grad = np.sum(self.de_lst[m], axis = 0,keepdims = True)
                self.bv_lst[m] = grad*lr + self.bv_lst[m]*g
                self.b_lst[m] -= self.bv_lst[m]
             self.h_lst = []
             self.z_lst = []
             self.de_lst = []
                
            if i%1000 ==0:
                full_preds = self.predict(x)
                hinge_loss = np.mean(np.maximum(0, 1 - y*full_preds))
                print(f"The hinge loss at {i} iteration is",hinge_loss)
                self.loss.append(hinge_loss)
                
            if i%1000 == 0:
                lr*=0.8

    def predict(self, x):
        h = x @ self.w_lst[0] + self.b_lst[0]
        z = self.relu(h)
        for j in range(1, len(self.w_lst)):
            h = z @ self.w_lst[j] + self.b_lst[j]
            if j != len(self.w_lst) - 1:
                z = self.relu(h)
        return h

    def plot(self):
        plt.plot(self.loss)
        plt.xlabel("iterations (x1000)")
        plt.ylabel("Hinge loss")
        plt.grid(True)
        plt.title("Loss graph utilising Hinge loss")
        plt.show()

model = dynamic_SVM(10, 64, 1, 3)
initial = time.perf_counter()
model.train(x, y,1024)
final = time.perf_counter()
print(f"The time taken is {final - initial:.2f} seconds")
model.plot()
