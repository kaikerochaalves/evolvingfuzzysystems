# evolvingfuzzysystems

Project description

Powered by Kaike Sa Teles Rocha Alves

evolvingfuzzysystems is a package that contains stated evolving Fuzzy Systems in the context of machine learning since the proposed eTS up to ePL-KRLS-DISCO 

    Website: kaikealves.weebly.com
    Documentation: https://doi.org/10.1016/j.asoc.2021.107764
    Email: kaikerochaalves@outlook.com
    Source code: https://github.com/kaikerochaalves/evolvingfuzzysystems.git

It provides:

    ePL-KRLS-DISCO, ePL+, eMG, ePL, Simpl_eTS, exTS, and eTS


Code of Conduct

evolvingfuzzysystems is a library developed by Kaike Alves. Please read the Code of Conduct for guidance.

Call for Contributions

The project welcomes your expertise and enthusiasm!

Small improvements or fixes are always appreciated. If you are considering larger contributions to the source code, please contact by email first.

To install the library use the command: 

    pip install evolvingfuzzysystems

To import the ePL-KRLS-DISCO, simply type the command:

    from evolvingfuzzysystems.eFS import ePL_KRLS_DISCO

To import the ePL+, simply type:

    from evolvingfuzzysystems.eFS import ePL_plus

To import the eMG, type:

    from evolvingfuzzysystems.eFS import eMG

To import the ePL, type:

    from evolvingfuzzysystems.eFS import ePL

To import the Simpl_eTS, type:

    from evolvingfuzzysystems.eFS import Simpl_eTS

To import the exTS, type:

    from evolvingfuzzysystems.eFS import exTS

To import the eTS, type:

    from evolvingfuzzysystems.eFS import eTS

Once you imported the libraries, you can use functions fit and predict. For example:

    from evolvingfuzzysystems.eFS import ePL_KRLS_DISCO
    model = ePL_KRLS_DISCO()
    model.fit(X_train, y_train)
    y_pred = model.predict(y_test)

To evolve the model, type:

    model.evolve(X_new, y_new)

The fuzzy models are quite fast, but the genetic and ensembles are still a bit slow. If you think you can contribute to this project regarding the code, speed, etc., please, feel free to contact me and to do so.
