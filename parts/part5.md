# socket
你需要为你的程序添加一个socket接口，别人会往socket中传入一整个矩阵，然后你需要把这个矩阵交给model类的forward方法，并把得到的返回值再通过socket发回去

你需要自行设计socket接收的参数，因为你要把收到的一堆数字转成一个二维矩阵，如果没有矩阵的长宽那显然有很多种不同的转法

你可以写一个简单的程序验证下socket收发是否正确