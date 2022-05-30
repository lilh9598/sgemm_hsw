/*
        k       -> x0
        a       -> x1
        b       -> x2
        c       -> x3
        ldc     -> x4
*/
.global kernel_12x8
kernel_12x8:
        stp     d8, d9, [sp, -48]!
        stp     d10, d11, [sp, 16]
        str     d12, [sp, 32]
        and     x5, x0, #1
        sub     x0, x0, x5
        lsl     x4, x4, 2
        add     x11, x3, x4
        add     x10, x11, x4
        add     x9, x10, x4
        add     x8, x9, x4
        add     x7, x8, x4
        add     x6, x7, x4
        add     x12, x6, x4
        ld1     {v5.4s, v6.4s, v7.4s}, [x3]
        ld1     {v8.4s, v9.4s, v10.4s}, [x11]
        ld1     {v11.4s, v12.4s, v13.4s}, [x10]
        ld1     {v14.4s, v15.4s, v16.4s}, [x9]
        ld1     {v17.4s, v18.4s, v19.4s}, [x8]
        ld1     {v20.4s, v21.4s, v22.4s}, [x7]
        ld1     {v23.4s, v24.4s, v25.4s}, [x6]
        ld1     {v26.4s, v27.4s, v28.4s}, [x12]
        cbz     x0, .end
.loop.2:
        ld1     {v0.4s, v1.4s}, [x2], #32
        ld1     {v2.4s, v3.4s, v4.4s}, [x1], #48

        fmla    v5.4s, v2.4s, v0.s[0]
        fmla    v8.4s, v2.4s, v0.s[1]
        fmla    v11.4s, v2.4s, v0.s[2]
        fmla    v14.4s, v2.4s, v0.s[3]

        fmla    v6.4s, v3.4s, v0.s[0]
        fmla    v9.4s, v3.4s, v0.s[1]
        fmla    v12.4s, v3.4s, v0.s[2]
        fmla    v15.4s, v3.4s, v0.s[3]

        fmla    v7.4s, v4.4s, v0.s[0]
        fmla    v10.4s, v4.4s, v0.s[1]
        fmla    v13.4s, v4.4s, v0.s[2]
        fmla    v16.4s, v4.4s, v0.s[3]

        fmla    v17.4s, v2.4s, v1.s[0]
        fmla    v20.4s, v2.4s, v1.s[1]
        fmla    v23.4s, v2.4s, v1.s[2]
        fmla    v26.4s, v2.4s, v1.s[3]

        fmla    v18.4s, v3.4s, v1.s[0]
        fmla    v21.4s, v3.4s, v1.s[1]
        fmla    v24.4s, v3.4s, v1.s[2]
        fmla    v27.4s, v3.4s, v1.s[3]

        fmla    v19.4s, v4.4s, v1.s[0]
        fmla    v22.4s, v4.4s, v1.s[1]
        fmla    v25.4s, v4.4s, v1.s[2]
        fmla    v28.4s, v4.4s, v1.s[3]

        // loop 2
        ld1     {v0.4s, v1.4s}, [x2], #32
        ld1     {v29.4s, v30.4s, v31.4s}, [x1], #48

        fmla    v5.4s, v29.4s, v0.s[0]
        fmla    v8.4s, v29.4s, v0.s[1]
        fmla    v11.4s, v29.4s, v0.s[2]
        fmla    v14.4s, v29.4s, v0.s[3]

        fmla    v6.4s, v30.4s, v0.s[0]
        fmla    v9.4s, v30.4s, v0.s[1]
        fmla    v12.4s, v30.4s, v0.s[2]
        fmla    v15.4s, v30.4s, v0.s[3]

        fmla    v7.4s, v31.4s, v0.s[0]
        fmla    v10.4s, v31.4s, v0.s[1]
        fmla    v13.4s, v31.4s, v0.s[2]
        fmla    v16.4s, v31.4s, v0.s[3]

        fmla    v17.4s, v29.4s, v1.s[0]
        fmla    v20.4s, v29.4s, v1.s[1]
        fmla    v23.4s, v29.4s, v1.s[2]
        fmla    v26.4s, v29.4s, v1.s[3]

        fmla    v18.4s, v30.4s, v1.s[0]
        fmla    v21.4s, v30.4s, v1.s[1]
        fmla    v24.4s, v30.4s, v1.s[2]
        fmla    v27.4s, v30.4s, v1.s[3]

        fmla    v19.4s, v31.4s, v1.s[0]
        fmla    v22.4s, v31.4s, v1.s[1]
        fmla    v25.4s, v31.4s, v1.s[2]
        fmla    v28.4s, v31.4s, v1.s[3]

        sub     x0, x0, 2
        cbnz    x0, .loop.2
.loop.1:
        cbz     x5, .end
        ld1     {v0.4s, v1.4s}, [x2]
        ld1     {v2.4s, v3.4s, v4.4s}, [x1]

        fmla    v5.4s, v2.4s, v0.s[0]
        fmla    v8.4s, v2.4s, v0.s[1]
        fmla    v11.4s, v2.4s, v0.s[2]
        fmla    v14.4s, v2.4s, v0.s[3]

        fmla    v6.4s, v3.4s, v0.s[0]
        fmla    v9.4s, v3.4s, v0.s[1]
        fmla    v12.4s, v3.4s, v0.s[2]
        fmla    v15.4s, v3.4s, v0.s[3]

        fmla    v7.4s, v4.4s, v0.s[0]
        fmla    v10.4s, v4.4s, v0.s[1]
        fmla    v13.4s, v4.4s, v0.s[2]
        fmla    v16.4s, v4.4s, v0.s[3]

        fmla    v17.4s, v2.4s, v1.s[0]
        fmla    v20.4s, v2.4s, v1.s[1]
        fmla    v23.4s, v2.4s, v1.s[2]
        fmla    v26.4s, v2.4s, v1.s[3]

        fmla    v18.4s, v3.4s, v1.s[0]
        fmla    v21.4s, v3.4s, v1.s[1]
        fmla    v24.4s, v3.4s, v1.s[2]
        fmla    v27.4s, v3.4s, v1.s[3]

        fmla    v19.4s, v4.4s, v1.s[0]
        fmla    v22.4s, v4.4s, v1.s[1]
        fmla    v25.4s, v4.4s, v1.s[2]
        fmla    v28.4s, v4.4s, v1.s[3]
.end:
        st1     {v5.4s, v6.4s, v7.4s}, [x3]
        st1     {v8.4s, v9.4s, v10.4s}, [x11]
        st1     {v11.4s, v12.4s, v13.4s}, [x10]
        st1     {v14.4s, v15.4s, v16.4s}, [x9]
        st1     {v17.4s, v18.4s, v19.4s}, [x8]
        st1     {v20.4s, v21.4s, v22.4s}, [x7]
        st1     {v23.4s, v24.4s, v25.4s}, [x6]
        st1     {v26.4s, v27.4s, v28.4s}, [x12]
        ldr     d12, [sp, 32]
        ldp     d10, d11, [sp, 16]
        ldp     d8, d9, [sp], 48
        ret
