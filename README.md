# SART: Sign-Absolute Reformulation Theory for Binary Variable Reduction in Neural Network Verification



This repository contains the official implementation of **SART**, accepted at **OOPSLA 2026**.  
The code is currently being cleaned and prepared for public release. The full implementation, including verifiers (LayerABS, Incomplete-LayerABS) and benchmark scripts, will be available soon.

Stay tuned! ⭐ **Watch** this repo to get notified when the code is released.

---

## 📄 Paper

**Title:** *SART: Sign-Absolute Reformulation Theory for Binary Variable Reduction in Neural Network Verification*  
**Conference:** OOPSLA 2026 (Proceedings of the ACM on Programming Languages)

**Authors:**  
- Jin Xu (Tongji University)
- Miaomiao Zhang (Tongji University)
- Bowen Du (Tongji University)

**Abstract:**  
Complete formal verification of neural networks is crucial for their deployment in safety-critical domains. A key bottleneck stems from encoding complexity: traditional methods assign one binary variable per unstable ReLU neuron. We propose the Sign-Absolute Reformulation Theory (SART), which fundamentally breaks the conventional one-to-one mapping between unstable neurons and binary variables by establishing formal reducibility criteria. This allows for finer-grained modeling, where each unstable neuron corresponds on average to fewer than one binary variable, thereby reducing verification complexity at its source.

Based on SART, we derive a theoretical lower bound on the number of binary variables required for complete verification and, under the assumption that $P \neq NP$, prove that variables in the final layer can be compressed by 50%, while the number of variables in intermediate layers cannot be further reduced. To overcome the apparent “last-layer-only” limitation, we recast verification as a sequential process and, crucially, show that the gain lifts to the entire network: LayerABS, a SART-based progressive tightening verifier, iteratively treats intermediate layers as temporary final layers and propagates tight bounds that shrink the global search space and binary-variable counts.

Furthermore, we reveal a structural law influencing verification complexity: when the signs of weights of unstable neurons satisfy numerical symmetry, with positive and negative weights equal or differing by at most one, the worst-case verification complexity achieves the theoretical optimum, offering theoretical guidance for the design of verification-friendly architectures.

As a general-purpose underlying encoding, the value of SART is independent of specific algorithms. To comprehensively evaluate its effectiveness, we first evaluate the abstraction-free SART encoding, and then integrate it with abstraction techniques to construct the complete verifier LayerABS and its incomplete variant Incomplete-LayerABS. Across benchmarks, our methods surpass state-of-the-art baselines, validating SART's practical impact.

---

## 🔧 Code Status

The code is under active cleanup and will be published in this repository. It includes:

- Core implementation of SART encoding
- Complete verifier **LayerABS**
- Incomplete verifier **Incomplete-LayerABS**
- Scripts to reproduce all experiments from the paper
- Benchmark instances and instructions

Please **star** ⭐ this repository to stay updated.

---

## 📚 Citation

If you find this work useful for your research, please cite our paper (bibtex will be added once the proceedings are published):

```bibtex
@inproceedings{SART2026,
  title     = {SART: Sign-Absolute Reformulation Theory for Binary Variable Reduction in Neural Network Verification},
  author    = {Jin Xu, Miaomiao Zhang and Bowen Du},
  booktitle = {Proceedings of the ACM on Programming Languages (OOPSLA)},
  year      = {2026}
}
```


---

## 📝 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
