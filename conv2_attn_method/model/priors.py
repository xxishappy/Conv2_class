#基于先验知识引导的注意力生成
##rect:矩形注意力：框内为1，框外为0
##gaussian:高斯注意力：框中心为原点，向四周高斯分布
##polar:极坐标注意力：任取框四条边中任一边中点为极点，向框内投影

#input:1.经过同步增强的mask:(B,1,H,W)，2.目标stage大的分辨率（H,W)
#output:A:注意力图,(B,1,H,W)-->对外接口PriorGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F

#mask_to_box,把mask反解为框，更精确，能够避免阶梯状伪影
def masks_to_boxes(mask,eps=0.5):
    '''
    将(B,1,H,W)的二值mask转化为(B,4)归一化xyxy框,值域为[0,1]
    当mask为空时,退回原图,等价于“无空间先验约束"
    #mask:dataset输出的mask
    #eps:决定前景后景的像素取值
    #input:mask(B,1,H,1)
    #output:box(B,4)
    '''
    if mask.dim()==4:
        mask=mask[:,0]#C只取第0个元素，即所有图片只取第0个通道
    B,H,W=mask.shape

    m=mask>eps  #大于0.5算前景，但mask按最近邻变换，只会有0,1
    dev=mask.device

    #找到有mask的行和列
    any_x=m.any(dim=1) #（B,W)每列是否有前景 #dim=X xw维被压缩 
    any_y=m.any(dim=2) #（B,H)每行是否有前景

    #创建坐标表，行号[0,1,...,W-1],列号[0,1,...,W-1],复制batch份    
    xs=torch.arange(W,device=dev).unsqueeze(0).expand(B,W) #扩展Batch维，生成列号
    ys=torch.arange(H,device=dev).unsqueeze(0).expand(B,H) #扩展Batch维，生成行号

    #准备极端值
    big_x=torch.full_like(xs,W) #使用W填充xs
    big_y=torch.full_like(ys,H)
    neg_x=torch.full_like(xs,-1)
    neg_y=torch.full_like(ys,-1)

    #找最小外接框的左上角坐标和右下角坐标
    #where(条件，真值取xs,假值取big_x)
    x1=torch.where(any_x,xs,big_x).min(dim=1).values.float() #最小的x值
    x2=torch.where(any_x,xs,neg_x).max(dim=1).values.float()+1.0 #最大的x值

    y1=torch.where(any_y,ys,big_y).min(dim=1).values.float() #最小的y值
    y2=torch.where(any_y,ys,neg_y).max(dim=1).values.float()+1.0 #最大的y值

    #使用空的mask做最低选择，空mask使用全图[0,0,W,H]
    empty=~m.flatten(1).any(dim=1)
    x1=torch.where(empty,torch.zeros_like(x1),x1)
    y1=torch.where(empty,torch.zeros_like(y1),y1)
    x2 = torch.where(empty, torch.full_like(x2, float(W)), x2)        
    y2 = torch.where(empty, torch.full_like(y2, float(H)), y2)

    box = torch.stack([x1 / W, y1 / H, x2 / W, y2 / H], dim=1)
    return box.clamp(0.0, 1.0)

#判断像素离目标(极点，box中心点)的距离
def grid(h,w,device,dtype):
    '''
    input:h,w:目标特征图的高和宽;device,dtype:需与box一致
    output:grid_x:(1,h,w)每个像素的x坐标,grid_y:(1,h,w)每个像素的y坐标
    '''
    #寻找行和列的中心点
    y_coords=(torch.arange(h,device=device,dtype=dtype)+0.5)/h #+0.5：取像素中心
    x_coords=(torch.arange(w,device=device,dtype=dtype)+0.5)/w

    #生成二维坐标网格
    #meshgrid:将两个一维数组扩展为二维网格
    grid_y,grid_x=torch.meshgrid(y_coords,x_coords,indexing="ij")

    #增加batch维度
    gx=grid_x.unsqueeze(0)
    gy=grid_y.unsqueeze(0)
    return gx,gy


#rect:矩形先验注意力
class RectPrior(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self,box,size):
        h,w=size
        gx,gy=grid(h,w,box.device,box.dtype)
        x1,y1,x2,y2=[t.view(-1,1,1) for t in box.unbind(dim=1)]
        #处于坐标内的像素
        inside=(gx>=x1)&(gx<x2)&(gy>=y1)&(gy<y2)
        return inside.to(box.dtype).unsqueeze(1)

#gaussian:高斯先验注意力
class GaussianPrior(nn.Module):
    def __init__(self,sigma_scale=0.5,normalize=True):
        super().__init__()
        self.normalize=normalize
        self.register_buffer("raw_sigma",torch.tensor(0.5))
    
    def forward(self,box,size):
        h,w=size
        gx,gy=grid(h,w,box.device,box.dtype)

        #找框的中线点和框的长和宽
        x1,y1,x2,y2=box.unbind(dim=1)
        cx=((x1+x2)*0.5).view(-1,1,1)#将维度变为(B,1,1),-1表示自动计算
        cy=((y1+y2)*0.5).view(-1,1,1)
        bw=(x2-x1).clamp_min(1e-3).view(-1,1,1) 
        bh=(y2-y1).clamp_min(1e-3).view(-1,1,1)

        s=F.softplus(self.raw_sigma).clamp(0.05,3.0)

        dx=(gx-cx)/(s*bw)
        dy=(gy-cy)/(s*bh)
        a=torch.exp(-0.5*(dx*dx+dy*dy))

        if self.normalize:
            a=a/a.amax(dim=(-2,-1),keepdim=True).clamp_min(1e-6)
        return a.unsqueeze(1) #(B,h,w) -> (B,1,h,w)

#polar:极坐标先验
class PolarPrior(nn.Module):
    '''
    四个极点p,均向框内投影:
    1.up:p=((x1+x2)/2,y1);2.down:p=((x1+x2)/2,y2);3.left:p=(x1,(y1+y2)/2);right:p=(x2,(y1+y2)/2)
    
    极点选择方式：
    1.四个极点固定其中一个
    2."random":训练时每个样本随机抽一个方向(只算一次,快),验证时四个极点的注意力取平均

    sigma_theta控制锥形张角
    '''
    POLES=("up","down","left","right")
    def __init__(self,pole="random",sigma_r=0.5,sigma_theta=0.5):
        super().__init__()
        self.pole=pole
        def raw(v,lo=1e-4):
            return torch.tensor(float(v)).expm1().clamp_min(lo).log()
        sr,st=raw(sigma_r),raw(sigma_theta)
       
        self.register_buffer("raw_sigma_r",sr)
        self.register_buffer("raw_sigma_theta",st)

    def single(self,box,size,pole):
        h,w=size
        gx,gy=grid(h,w,box.device,box.dtype)

        #找框的长和宽,及中心坐标
        x1, y1, x2, y2 = [t.view(-1, 1, 1) for t in box.unbind(dim=1)]
        bw = (x2 - x1).clamp_min(1e-3)
        bh = (y2 - y1).clamp_min(1e-3)
        cx, cy = (x1 + x2)*0.5, (y1 + y2)*0.5

        #确定极点(px,py)和法向量(nx,ny)
        if pole=="up":
            px,py,nx,ny=cx,y1,0.0,1.0
        elif pole=="down":
            px,py,nx,ny=cx,y2,0.0,-1.0
        elif pole=="left":
            px,py,nx,ny=x1,cy,1.0,0.0
        elif pole=="right":
            px,py,nx,ny=x2,cy,-1.0,0.0
        else:
            raise ValueError(f"未知pole:{pole}")
        
        #像素相对极点的偏移，用box宽高归一化
        dx=(gx-px)/bw
        dy=(gy-py)/bh
        r=torch.sqrt(dx*dx+dy*dy+1e-12) #加1e-12方式除0导出空值

        #极点到box最远角点的距离
        r_max=(1.0**2+0.5**2)**0.5
        r_hat=(r/r_max).clamp(0.0,4.0)

        #夹角theta:cos(theta)=(向量*法向量)/|向量|
        cos_t=(dx*nx+dy*ny)/r.clamp_min(1e-6)
        theta=torch.acos(cos_t.clamp(-1.0+1e-6,1.0-1e-6))

        sigma_r=F.softplus(self.raw_sigma_r).clamp(0.05,3.0)
        sigma_t=F.softplus(self.raw_sigma_theta).clamp(0.05,3.14)

        #峰值在极点r=0，沿法向量呈锥形衰减
        a=torch.exp(-0.5*(r_hat/sigma_r)**2)*torch.exp(-0.5*(theta/sigma_t)**2)

        return a

    def forward(self,box,size,return_pole_weight=False):
        if self.pole=="random":
            #训练时，随机挑选
            if self.training:
                B=box.shape[0]
                h,w=size

                #每个样本随机选择一个极点
                pick=torch.randint(len(self.POLES),(B,), device=box.device)

                #为每个样本计算他被选中的方向
                a=box.new_zeros((B,h,w))
                for k,p in enumerate(self.POLES):
                    idx=(pick==k).nonzero(as_tuple=True)[0]
                    if idx.numel()>0:
                        a[idx]=self.single(box[idx],size,p)
                    
                wt=torch.bincount(pick,minlength=len(self.POLES)).to(box.dtype)/max(B,1)
            else:
                a=0.0
                for p in self.POLES:
                    a=a+self.single(box,size,p)
                a=a/len(self.POLES)
                wt= torch.full((len(self.POLES),), 1.0 / len(self.POLES),device=box.device, dtype=box.dtype)
        else:
            a=self.single(box,size,self.pole)
            wt=None
        a=a/a.amax(dim=(-2,-1),keepdim=True).clamp_min(1e-6)
        a=a.unsqueeze(1)
        return (a,wt) if return_pole_weight else a
              
#接口
class PriorGenerator(nn.Module):
    '''
    prior_type可选:
        "rect"     框内1框外0
        "gaussian" 框中心高斯
        "polar"    极坐标,方向由pole参数决定
        "none"     不加先验(baseline)

    极坐标的方向通过pole参数传入:
        PriorGenerator("polar", pole="up")       固定自上边中点向下
        PriorGenerator("polar", pole="down")     固定自下边中点向上
        PriorGenerator("polar", pole="left")     固定自左边中点向右
        PriorGenerator("polar", pole="right")    固定自右边中点向左
        PriorGenerator("polar", pole="random")   训练随机选方向,eval四方向平均

    input:mask(B,1,H,W)增强后的二值ROI;size(h,w)目标stage分辨率
    output:A(B,1,h,w);prior_type="none"时返回None
    pipeline:mask--> box --> Attention Map A
    '''
  
    def __init__(self,prior_type="rect",**kw):
        super().__init__()
        self.prior_type=prior_type
        if prior_type=="rect":
            self.gen=RectPrior(**kw)
        elif prior_type=="gaussian":
            self.gen=GaussianPrior(**kw)
        elif prior_type=="polar":
            self.gen=PolarPrior(**kw)
        elif prior_type=="none":
            self.gen=None
        else:
            raise ValueError(f"未知prior_type:{prior_type};可选rect/gaussian/polar/none")
    
    def forward(self,mask,size):
        if self.gen is None:
            return None
        box=masks_to_boxes(mask)
        return self.gen(box,size)

    def from_box(self, box, size):

        if self.gen is None:
            return None
        return self.gen(box, size)
