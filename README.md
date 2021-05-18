# Spectral Perturbation Meets Incomplete Multi-view Data
This repo hosts the code for paper "Spectral Perturbation Meets Incomplete Multi-view Data, in IJCAI-2019". Beyond [_multi-view clustering_](https://github.com/cswanghao/Multi-view-Clustering), the paper tackles _incomplete multi-view clustering_ (a.k.a. partial multi-view clustering), where a number of data instances are missing in certain views.

The proposed method shows a strong link between spectral perturbation and incomplete multi-view clustering. The key idea is to transfer the missing problem from data matrix to similarity matrix and reduce the spectral perturbation risk among different views while balancing all views to learn a consensus representation for the final clustering results.

The repo also hosts some baseline systems as we compared in the paper. We would like to thank the authors of the baseline systems for their codes. _If any baseline systems cannot be licensed freely here, please drop me an email, so we can remove it from the collection._

If you find this repo useful, please kindly cite the paper below.

    @inproceedings{wang2019spectral,
      title={Spectral Perturbation Meets Incomplete Multi-view Data},
      author={Wang, Hao and Zong, Linlin and Liu, Bing and Yang, Yan and Zhou, Wei},
      booktitle={Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI)},
      pages={3677--3683},
      year={2019}
    }
