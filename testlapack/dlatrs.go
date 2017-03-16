// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/gonum/floats"
)

type Dlatrser interface {
	Dlatrs(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, normin bool, n int, a []float64, lda int, x []float64, cnorm []float64) (scale float64)
}

func DlatrsTest(t *testing.T, impl Dlatrser) {
	rnd := rand.New(rand.NewSource(1))
	for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
		for _, trans := range []blas.Transpose{blas.Trans, blas.NoTrans} {
			for _, diag := range []blas.Diag{blas.Unit, blas.NonUnit} {
				for _, normin := range []bool{true, false} {
					for _, n := range []int{0, 1, 2, 3, 4, 5, 10, 19} {
						for _, lda := range []int{n, 2*n + 1} {
							lda = max(1, lda)
							for _, imat := range []int{11, 12} {
								testDlatrs(t, impl, uplo, trans, diag, normin, n, lda, imat, rnd)
							}
						}
					}
				}
			}
		}
	}
}

func testDlatrs(t *testing.T, impl Dlatrser, uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, normin bool, n, lda int, imat int, rnd *rand.Rand) {
	const (
		dlamchE = 1.0 / (1 << 53)
		dlamchP = 2 * dlamchE
		dlamchB = 2
	)
	smlnum := math.Nextafter((4 / math.MaxFloat64), 0)
	bignum := (1 - dlamchP) / smlnum

	a := nanSlice(n * lda)
	switch uplo {
	case blas.Upper:
		for i := 0; i < n-1; i++ {
			for j := i + 1; j < n; j++ {
				a[i*lda+j] = 2*rnd.Float64() - 1
			}
		}
	case blas.Lower:
		for i := 1; i < n; i++ {
			for j := 0; j < i; j++ {
				a[i*lda+j] = 2*rnd.Float64() - 1
			}
		}
	default:
		panic("bad uplo")
	}
	if diag == blas.NonUnit {
		// Give the diagonal norm 2 to make it well-conditioned.
		for i := 0; i < n; i++ {
			if rnd.Intn(2) == 0 {
				a[i*lda+i] = 2
			} else {
				a[i*lda+i] = -2
			}
		}
	}

	// Set the right hand side so that the largest value is bignum.
	b := make([]float64, n)
	for i := range b {
		b[i] = 2*rnd.Float64() - 1
	}
	var bmax float64
	for _, v := range b {
		vabs := math.Abs(v)
		if vabs > bmax {
			bmax = vabs
		}
	}
	bscal := bignum / math.Max(1, bmax)
	blas64.Scal(n, bscal, blas64.Vector{1, b})

	want := make([]float64, len(b))
	copy(want, b)

	cnorm := make([]float64, n)

	scale := impl.Dlatrs(uplo, trans, diag, false, n, a, lda, b, cnorm)

	blas64.Scal(n, scale, blas64.Vector{1, want})
	blas64.Trmv(trans, blas64.Triangular{n, lda, a, uplo, diag}, blas64.Vector{1, b})

	if !floats.EqualApprox(b, want, 1e-14) {
		t.Errorf("Case n=%v,lda=%v,trans=%v,uplo=%v,diag=%v: unexpected result when scale=%v\n%v\n%v",
			n, lda, trans, uplo, diag, scale, b, want)
	}
}
