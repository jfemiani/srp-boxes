import matplotlib.pyplot as plt
import pathlib
import pickle

def plot_rgb(stack):
    plt.imshow(stack[:3].transpose(1,2,0).clip(0,1))    
    plt.xlim(0, stack.shape[1])
    plt.ylim(0, stack.shape[2])
    
def plot_lidar(stack, alpha=1): 
    pseudo = stack[4:7]
    pseudo = pseudo
    pseudo = sigmoid(pseudo.transpose(1,2,0))
    #pseudo -= pseudo.min()
    #pseudo /= pseudo.max()  
    alpha_ = pseudo.max(2)[...,None]*alpha
    pseudo = np.concatenate([pseudo, alpha_], axis=2)          
    plt.imshow(pseudo)
    plt.xlim(0, stack.shape[1])
    plt.ylim(0, stack.shape[2])
    
def _set_grid_spacing(minor, major):
    from matplotlib.ticker import MultipleLocator
    ax = gca()
    ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_minor_locator(MultipleLocator(minor))
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.grid(which='major')
    ax.grid(which='minor', linestyle='--')

def load_stack(path):
    if isinstance(sample, pathlib.PosixPath):
        path = path.as_posix()
    with open(path, 'rb') as handle:
        p = pickle.load(handle)
    return np.concatenate((p.rgb, p.volumetric))
    
    
    
def plot3(stack, major=None, minor=None):
    if major is None:
        major = stack.shape[1]//2
    if minor is None:
        minor = major // 4
        
    subplot(131); 
    plot_rgb(stack); 
    _set_grid_spacing(minor, major); 
    title('rgb')
    
    subplot(132); 
    plot_rgb(stack); 
    plot_lidar(stack); 
    set_grid_spacing(minor, major);
    
    title('mixed')
    
    subplot(133, facecolor='black'); 
    plot_lidar(stack); 
    set_grid_spacing(minor, major); 
    title('lidar'); 