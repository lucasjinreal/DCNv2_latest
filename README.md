# DCNv2 latest

- Add support for pytorch1.11 (may be not backward-compatible). 
- Test on ubuntu20.04, python3.8(conda), cuda_11.4

It was confirmed that pytorch1.11 worked, but not compatible with previous pytorch version. If you want pytorch1.10 or earlier, please using pytorch1.6 branch, or using last git commit.

It's suggested using latest stable pytorch 1.11 to start your project.


## Install

```bash
$ python3 setup.py build develop
```

## Updates

- **2021.03.24**: It was confirmed PyTorch 1.8 is OK with master branch, feel free to use it.
- **2021.02.18**: Happy new year! PyTorch 1.7 finally supported on master branch! **for lower version theoretically also works, if not, pls fire an issue to me!**.
- **2020.09.23**: Now master branch works for pytorch 1.6 by default, for older version you gonna need separated one.
- **2020.08.25**: Check out pytorch1.6 branch for pytorch 1.6 support, you will meet an error like `THCudaBlas_Sgemv undefined` if you using pytorch 1.6 build master branch. master branch now work for pytorch 1.5;

## Contact

If you have any question, please using this platform post questions: http://t.manaai.cn
