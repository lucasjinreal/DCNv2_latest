# DCNv2 latest



Since DCN is used in many models and performance well but in industry this op support is not very well. Including pytorch, onnx, tensorrt etc. This repo is make DCNv2 available at all versions in pytorch.

![image-20210217183103121](https://gitee.com/jinfagang/picbed/raw/master/img/image-20210217183103121.png)

Pytorch 1.7 inferenced in CenterNet-DLA model. It works on Pytorch 1.7 so that you can use it in your RTX 30 series cards.



## Updates

- **2021.03.24**: It was confirmed PyTorch 1.8 is OK with master branch, feel free to use it.
- **2021.02.18**: Happy new year! PyTorch 1.7 finally supported on master branch! **for lower version theoretically also works, if not, pls fire an issue to me!**.
- **2020.09.23**: Now master branch works for pytorch 1.6 by default, for older version you gonna need separated one.
- **2020.08.25**: Check out pytorch1.6 branch for pytorch 1.6 support, you will meet an error like `THCudaBlas_Sgemv undefined` if you using pytorch 1.6 build master branch. master branch now work for pytorch 1.5;



## Contact

If you have any question, welcome add my wechat: `jintianiloveu` for communication.
