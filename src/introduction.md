<p align="center">
<img width="150px" src="https://user-images.githubusercontent.com/18649508/141604792-b256023d-c751-46d8-bab5-29a207d714ba.png"/>
</p>

<hr/>

**PERSIA** (**P**arallel r**E**commendation t**R**aining **S**ystem with hybr**I**d **A**cceleration)  is developed by [AI platform@Kuaishou Technology](https://www.kuaishou.com/en), collaborating with ETH. It is a PyTorch-based (the first public one to our best knowledge) system for training large scale deep learning recommendation models on commodity hardware. It is capable of training recommendation models with up to 100 trillion parameters. To the best of our knowledge, this is the largest model size in recommendation systems so far. Empirical study on public datasets indicate PERSIA's significant advantage over several other existing training systems in recommendation (see [benchmark](benchmark/index.md) for details). Its efficiency and robustness have also been validated by multiple applications with 100 million level DAU at Kuaishou. 

## In the News
* AI Engines in the "Short-video" Era: Eating 100 Trillion Parameters, Invited talk, Facebook, 2021.
* 单机训练速度提升 640 倍！独家解读快手商业广告模型 GPU 训练平台 PERSIA (In Chinese. Title: 640x Faster GPU Based Learning System for Ad Recommendation)
   * [[AI Front]](https://archive.is/2ii2L) [[中国日报]](https://archive.is/N8fK2) [[InfoQ]](https://archive.is/JESDU) [[CSDN]](https://archive.is/tpvkN) [[Tencent Cloud News]](https://archive.is/kLuaT) [[AcFun]](https://archive.md/vuPmb)
* 创新、平衡与大格局：快手商业化的慢与快 (In Chinese. Title: Innovation, Balance, and Big Picture: The Speed of Kwai Commercialization)
   * [[TechSir]](https://archive.is/EOQ18) [[China Daily]](https://archive.is/L2VJE) [[Sohu]](https://archive.is/aY66U)

## Links

* [GitHub Repository](https://github.com/PersiaML/PERSIA)
* [Tutorials](https://persiaml-tutorials.pages.dev/)
* [API documentation](https://persiaml.pages.dev/)

## Discussion

Feel free to join our [Telegram Group](https://t.me/joinchat/fLlD66VX8PQxMmJh) for discussion!  

## References

1. Xiangru Lian, Binhang Yuan, Xuefeng Zhu, Yulong Wang, Yongjun He, Honghuan Wu, Lei Sun, Haodong Lyu, Chengjun Liu, Xing Dong, Yiqiao Liao, Mingnan Luo, Congfei Zhang, Jingru Xie, Haonan Li, Lei Chen, Renjie Huang, Jianying Lin, Chengchun Shu, Xuezhong Qiu, Zhishan Liu, Dongying Kong, Lei Yuan, Hai Yu, Sen Yang, Ce Zhang, & Ji Liu. (2021). [Persia: A Hybrid System Scaling Deep Learning Based Recommenders up to 100 Trillion Parameters.](https://arxiv.org/abs/2111.05897)

2. Ji Liu & Ce Zhang. (2021). [Distributed Learning Systems with First-order Methods](https://arxiv.org/pdf/2104.05245).

## License

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
