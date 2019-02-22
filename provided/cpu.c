#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

void H(float *p, float *q, int N, int a) {
    int t = (1<<a);
    int i;
    for (i = 0; i < N; i++) {
        if ((i & t) == 0) {
            q[i] = (p[i] + p[i^t]) / sqrt(2.0);
        } else {
            q[i] = (p[i^t] - p[i]) / sqrt(2.0);
        }
    }
}

void X(float *p, float *q, int N, int a) {
    int t = (1<<a);
    int i;
    for (i = 0; i < N; i++) {
        q[i] = p[i^t];
    }
}

void Z(float *p, float *q, int N, int a) {
    int t = (1<<a);
    int i;
    for (i = 0; i < N; i++) {
        if ((i & t) > 0) {
            q[i] = -p[i];
        } else {
            q[i] = p[i];
        }
    }
}

void CX(float *p, float *q, int N, int a, int b) {
    int ctrl = (1<<a);
    int t    = (1<<b);
    int i;
    for (i = 0; i < N; i++) {
        if ((i & ctrl) > 0) {
            q[i] = p[i^t];
        } else {
            q[i] = p[i];
        }
    }
}

void CZ(float *p, float *q, int N, int a, int b) {
    int ctrl = (1<<a);
    int t    = (1<<b);
    int i;
    for (i = 0; i < N; i++) {
        if ((i & ctrl) > 0 && (i & t) > 0) {
            q[i] = -p[i];
        } else {
            q[i] = p[i];
        }
    }
}

void CCX(float *p, float *q, int N, int a, int b, int c) {
    int ctrl0 = (1<<a);
    int ctrl1 = (1<<b);
    int t     = (1<<c);
    int i;
    for (i = 0; i < N; i++) {
        if ((i & ctrl0) > 0 && (i & ctrl1) > 0) {
            q[i] = p[i^t];
        } else {
            q[i] = p[i];
        }
    }
}

int main() {
    int n, m, N;
    float *P, *Q;
    scanf("%d%d", &n, &m);
    
    N = 1 << n;
    // P と Q を交互に使う
    P = malloc(sizeof(*P) * N);
    Q = malloc(sizeof(*P) * N);

    // P[0] のみ 1 で他は 0 で初期化
    memset(P, 0, sizeof(*P) * N);
    memset(Q, 0, sizeof(*Q) * N);
    P[0] = 1.0;

    char gate[4];
    int i;
    int a, b, c;
    for (i = 0; i < m; i++) {
        float *p, *q; // p を元に q に結果を書き込む
        if (i % 2 == 0) {
            p = P; q = Q;
        } else {
            p = Q; q = P;
        }
        scanf("%s", gate);
        // strcmp(s1, s2) は s1 と s2 が同じ文字列の時 0 を返す
        if (strcmp(gate, "H") == 0) {
            scanf("%d", &a);
            H(p, q, N, a);
        } else if (strcmp(gate, "X") == 0) {
            scanf("%d", &a);
            X(p, q, N, a);
        } else if (strcmp(gate, "Z") == 0) {
            scanf("%d", &a);
            Z(p, q, N, a);
        } else if (strcmp(gate, "CX") == 0) {
            scanf("%d%d", &a, &b);
            CX(p, q, N, a, b);
        } else if (strcmp(gate, "CZ") == 0) {
            scanf("%d%d", &a, &b);
            CZ(p, q, N, a, b);
        } else if (strcmp(gate, "CCX") == 0) {
            scanf("%d%d%d", &a, &b, &c);
            CCX(p, q, N, a, b, c);
        } else {
            printf("input error.\n");
            return 1;
        }
    }

    // 最大値とその位置を求める
    float *p;
    if (m % 2 == 0) p = P;
    else p = Q;
    float ans = 0;
    int idx = 0;
    for (i = 0; i < N; i++) {
        if (ans < p[i] * p[i]) {
            ans = p[i] * p[i];
            idx = i;
        }
    }

    printf("%d\n", idx);
    printf("%.8e\n", ans);
    
    return 0;
}
