import numpy as np
import time
from scipy.optimize import minimize
from sklearn.svm import LinearSVC

def predict_tsvm_cccp(X_labeled, y_labeled, X_unlabeled,
                      C=1.0, C_star_final=0.1, gamma=1.0, verbose=False):
    # Преобразуем метки: 0 -> -1, 1 -> 1
    y_l = np.where(y_labeled == 0, -1, 1)

    # Инициализация линейной SVM
    lin_svm = LinearSVC(C=C, loss='hinge', fit_intercept=True, random_state=0)
    lin_svm.fit(X_labeled, y_l)
    w0 = lin_svm.coef_.ravel()
    b0 = lin_svm.intercept_[0]

    # Балансировка классов
    unique, counts = np.unique(y_l, return_counts=True)
    class_weight = {lbl: len(y_l) / (2 * count) for lbl, count in zip(unique, counts)}

    def objective(theta, X_l, y_l, X_u, C, C_star, gamma, class_weight):
        d = X_l.shape[1]
        w, b = theta[:d], theta[-1]

        margin_l = y_l * (X_l @ w + b)
        weights = np.array([class_weight[int(lbl)] for lbl in y_l])
        loss_l = weights * np.maximum(0, 1 - margin_l)**2

        f_u = X_u @ w + b
        loss_u = np.exp(-5.0 * f_u**2)

        return 0.5 * np.dot(w, w) + C * loss_l.sum() + C_star * loss_u.sum()

    def gradient(theta, X_l, y_l, X_u, C, C_star, gamma, class_weight):
        d = X_l.shape[1]
        w, b = theta[:d], theta[-1]

        f_l = X_l @ w + b
        margin = y_l * f_l
        active = margin < 1

        weights = np.array([class_weight[int(lbl)] for lbl in y_l[active]])
        coeff_l = -2.0 * (1 - margin[active]) * y_l[active] * weights
        grad_w_l = C * (coeff_l[:, None] * X_l[active]).sum(axis=0)
        grad_b_l = C * coeff_l.sum()

        f_u = X_u @ w + b
        exp_val = np.exp(-5.0 * f_u**2)
        coeff_u = -2.0 * 5.0 * f_u * exp_val
        grad_w_u = C_star * (coeff_u[:, None] * X_u).sum(axis=0)
        grad_b_u = C_star * coeff_u.sum()

        grad_w = w + grad_w_l + grad_w_u
        grad_b = grad_b_l + grad_b_u
        return np.concatenate([grad_w, [grad_b]])

    # Аннилинг по C*
    C_star_seq = 2.0**(np.linspace(-10, 0, 10)) * C_star_final
    theta = np.hstack([w0, b0])

    start = time.time()
    for k, C_star in enumerate(C_star_seq, 1):
        res = minimize(
            objective, theta, jac=gradient,
            args=(X_labeled, y_l, X_unlabeled, C, C_star, gamma, class_weight),
            method='L-BFGS-B', options=dict(maxiter=150, gtol=1e-4, disp=False)
        )
        theta = res.x
        if verbose:
            print(f"  Шаг {k:2d}/10: C*={C_star:6.4f}, "
                  f"f={res.fun:8.3f}, ‖∇‖={np.linalg.norm(res.jac):7.4f}")
    end = time.time()

    if verbose:
        print(f"\n⏱ Время градиентного метода с аннилингом: {end - start:.4f} секунд")

    # Предсказания
    w_opt, b_opt = theta[:-1], theta[-1]
    y_pred = np.sign(X_unlabeled @ w_opt + b_opt)
    return np.where(y_pred == -1, 0, 1)
