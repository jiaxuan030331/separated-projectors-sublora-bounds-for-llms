"""Abstract linear algebra library.
This module defines a class hierarchy that implements a kind of "lazy"
matrix representation, called the ``LinearOperator``. It can be used to do
linear algebra with extremely large sparse or structured matrices, without
representing those explicitly in memory. Such matrices can be added,
multiplied, transposed, etc.
As a motivating example, suppose you want have a matrix where almost all of
the elements have the value one. The standard sparse matrix representation
skips the storage of zeros, but not ones. By contrast, a LinearOperator is
able to represent such matrices efficiently. First, we need a compact way to
represent an all-ones matrix::
    >>> import numpy as np
    >>> class Ones(LinearOperator):
    ...     def __init__(self, shape):
    ...         super(Ones, self).__init__(dtype=None, shape=shape)
    ...     def _matvec(self, x):
    ...         return np.repeat(x.sum(), self.shape[0])
Instances of this class emulate ``np.ones(shape)``, but using a constant
amount of storage, independent of ``shape``. The ``_matvec`` method specifies
how this linear operator multiplies with (operates on) a vector. We can now
add this operator to a sparse matrix that stores only offsets from one::
    >>> from scipy.sparse import csr_matrix
    >>> offsets = csr_matrix([[1, 0, 2], [0, -1, 0], [0, 0, 3]])
    >>> A = aslinearoperator(offsets) + Ones(offsets.shape)
    >>> A.dot([1, 2, 3])
    array([13,  4, 15])
The result is the same as that given by its dense, explicitly-stored
counterpart::
    >>> (np.ones(A.shape, A.dtype) + offsets.toarray()).dot([1, 2, 3])
    array([13,  4, 15])
Several algorithms in the ``scipy.sparse`` library are able to operate on
``LinearOperator`` instances.
"""

import warnings

import torch
import numpy as np
# import jax
__all__ = ['LinearOperator', 'aslinearoperator']


class LinearOperator(object):
    """Common interface for performing matrix vector products
    Many iterative methods (e.g. cg, gmres) do not need to know the
    individual entries of a matrix to solve a linear system A*x=b.
    Such solvers only require the computation of matrix vector
    products, A*v where v is a dense vector.  This class serves as
    an abstract interface between iterative solvers and matrix-like
    objects.
    To construct a concrete LinearOperator, either pass appropriate
    callables to the constructor of this class, or subclass it.
    A subclass must implement either one of the methods ``_matvec``
    and ``_matmat``, and the attributes/properties ``shape`` (pair of
    integers) and ``dtype`` (may be None). It may call the ``__init__``
    on this class to have these attributes validated. Implementing
    ``_matvec`` automatically implements ``_matmat`` (using a naive
    algorithm) and vice-versa.
    Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint``
    to implement the Hermitian adjoint (conjugate transpose). As with
    ``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or
    ``_adjoint`` implements the other automatically. Implementing
    ``_adjoint`` is preferable; ``_rmatvec`` is mostly there for
    backwards compatibility.
    Parameters
    ----------
    shape : tuple
        Matrix dimensions (M, N).
    matvec : callable f(v)
        Returns returns A * v.
    rmatvec : callable f(v)
        Returns A^H * v, where A^H is the conjugate transpose of A.
    matmat : callable f(V)
        Returns A * V, where V is a dense matrix with dimensions (N, K).
    dtype : dtype
        Data type of the matrix.
    rmatmat : callable f(V)
        Returns A^H * V, where V is a dense matrix with dimensions (M, K).
    Attributes
    ----------
    args : tuple
        For linear operators describing products etc. of other linear
        operators, the operands of the binary operation.
    ndim : int
        Number of dimensions (this is always 2)
    See Also
    --------
    aslinearoperator : Construct LinearOperators
    Notes
    -----
    The user-defined matvec() function must properly handle the case
    where v has shape (N,) as well as the (N,1) case.  The shape of
    the return type is handled internally by LinearOperator.
    LinearOperator instances can also be multiplied, added with each
    other and exponentiated, all lazily: the result of these operations
    is always a new, composite LinearOperator, that defers linear
    operations to the original operators and combines the results.
    More details regarding how to subclass a LinearOperator and several
    examples of concrete LinearOperator instances can be found in the
    external project `PyLops <https://pylops.readthedocs.io>`_.
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import LinearOperator
    >>> def mv(v):
    ...     return np.array([2*v[0], 3*v[1]])
    ...
    >>> A = LinearOperator((2,2), matvec=mv)
    >>> A
    <2x2 _CustomLinearOperator with dtype=float64>
    >>> A.matvec(np.ones(2))
    array([ 2.,  3.])
    >>> A * np.ones(2)
    array([ 2.,  3.])
    """

    ndim = 2

    def __new__(cls, *args, **kwargs):
        if cls is LinearOperator:
            # Operate as _CustomLinearOperator factory.
            return super(LinearOperator, cls).__new__(_CustomLinearOperator)
        else:
            obj = super(LinearOperator, cls).__new__(cls)

            if (type(obj)._matvec == LinearOperator._matvec
                    and type(obj)._matmat == LinearOperator._matmat):
                warnings.warn("LinearOperator subclass should implement"
                              " at least one of _matvec and _matmat.",
                              category=RuntimeWarning, stacklevel=2)

            return obj

    def __init__(self, dtype, shape):
        """Initialize this LinearOperator.
        To be called by subclasses. ``dtype`` may be None; ``shape`` should
        be convertible to a length-2 tuple.
        """
        if dtype is not None:
            dtype = np.dtype(dtype)

        shape = tuple(shape)
        if not isshape(shape):
            raise ValueError("invalid shape %r (must be 2-d)" % (shape,))

        self.dtype = np.dtype('float32') #force float 32
        self.shape = shape

    def _init_dtype(self):
        """Called from subclasses at the end of the __init__ routine.
        """
        if self.dtype is None:
            #v = np.zeros(self.shape[-1])
            self.dtype = np.dtype('float32')#self.matvec(v).dtype #force float 32

    def _matmat(self, X):
        """Default matrix-matrix multiplication handler.
        Falls back on the user-defined _matvec method, so defining that will
        define matrix multiplication (though in a very suboptimal way).
        """

        return torch.hstack([self.matvec(col.reshape(-1,1)) for col in X.T])

    def _matvec(self, x):
        """Default matrix-vector multiplication handler.
        If self is a linear operator of shape (M, N), then this method will
        be called on a shape (N,) or (N, 1) ndarray, and should return a
        shape (M,) or (M, 1) ndarray.
        This default implementation falls back on _matmat, so defining that
        will define matrix-vector multiplication as well.
        """
        return self.matmat(x.reshape(-1, 1))

    def matvec(self, x):
        """Matrix-vector multiplication.
        Performs the operation y=A*x where A is an MxN linear
        operator and x is a column vector or 1-d array.
        Parameters
        ----------
        x : {matrix, ndarray}
            An array with shape (N,) or (N,1).
        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (M,) or (M,1) depending
            on the type and shape of the x argument.
        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden
        _matvec method to ensure that y has the correct shape and type.
        """

        M,N = self.shape
        if x.shape != (N,) and x.shape != (N,1):
            raise ValueError('dimension mismatch')

        y = self._matvec(x)

        if x.ndim == 1:
            y = y.reshape(M)
        elif x.ndim == 2:
            y = y.reshape(M,1)
        else:
            raise ValueError('invalid shape returned by user-defined matvec()')

        return y

    def rmatvec(self, x):
        """Adjoint matrix-vector multiplication.
        Performs the operation y = A^H * x where A is an MxN linear
        operator and x is a column vector or 1-d array.
        Parameters
        ----------
        x : {matrix, ndarray}
            An array with shape (M,) or (M,1).
        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (N,) or (N,1) depending
            on the type and shape of the x argument.
        Notes
        -----
        This rmatvec wraps the user-specified rmatvec routine or overridden
        _rmatvec method to ensure that y has the correct shape and type.
        """
        M,N = self.shape

        if x.shape != (M,) and x.shape != (M,1):
            raise ValueError('dimension mismatch')

        y = self._rmatvec(x)

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N,1)
        else:
            raise ValueError('invalid shape returned by user-defined rmatvec()')

        return y

    def _rmatvec(self, x):
        """Default implementation of _rmatvec; defers to adjoint."""
        if type(self)._adjoint == LinearOperator._adjoint:
            # _adjoint not overridden, prevent infinite recursion
            raise NotImplementedError
        else:
            return self.T.matvec(x)

    def matmat(self, X):
        """Matrix-matrix multiplication.
        Performs the operation y=A*X where A is an MxN linear
        operator and X dense N*K matrix or ndarray.
        Parameters
        ----------
        X : {matrix, ndarray}
            An array with shape (N,K).
        Returns
        -------
        Y : {matrix, ndarray}
            A matrix or ndarray with shape (M,K) depending on
            the type of the X argument.
        Notes
        -----
        This matmat wraps any user-specified matmat routine or overridden
        _matmat method to ensure that y has the correct type.
        """

        if X.ndim != 2:
            raise ValueError('expected 2-d ndarray or matrix, not %d-d'
                             % X.ndim)

        if X.shape[0] != self.shape[1]:
            raise ValueError('dimension mismatch: %r, %r'
                             % (self.shape, X.shape))

        Y = self._matmat(X)
        return Y

    def rmatmat(self, X):
        """Adjoint matrix-matrix multiplication.
        Performs the operation y = A^H * x where A is an MxN linear
        operator and x is a column vector or 1-d array, or 2-d array.
        The default implementation defers to the adjoint.
        Parameters
        ----------
        X : {matrix, ndarray}
            A matrix or 2D array.
        Returns
        -------
        Y : {matrix, ndarray}
            A matrix or 2D array depending on the type of the input.
        Notes
        -----
        This rmatmat wraps the user-specified rmatmat routine.
        """

        if X.ndim != 2:
            raise ValueError('expected 2-d ndarray or matrix, not %d-d'
                             % X.ndim)

        if X.shape[0] != self.shape[0]:
            raise ValueError('dimension mismatch: %r, %r'
                             % (self.shape, X.shape))

        Y = self._rmatmat(X)
        return Y

    def _rmatmat(self, X):
        """Default implementation of _rmatmat defers to rmatvec or adjoint."""
        if type(self)._adjoint == LinearOperator._adjoint:
            return torch.hstack([self.rmatvec(col.reshape(-1, 1)) for col in X.T])
        else:
            return self.T.matmat(X)

    def __call__(self, x):
        return self*x

    def __mul__(self, x):
        if np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        return self.dot(x)

    def dot(self, x):
        """Matrix-matrix or matrix-vector multiplication.
        Parameters
        ----------
        x : array_like
            1-d or 2-d array, representing a vector or matrix.
        Returns
        -------
        Ax : array
            1-d or 2-d array (depending on the shape of x) that represents
            the result of applying this linear operator on x.
        """
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(self, x)
        elif np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            else:
                raise ValueError('expected 1-d or 2-d array or matrix, got %r'
                                 % x)

    def __matmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__mul__(other)

    def __rmatmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__rmul__(other)

    def __rmul__(self, x):
        if np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            return NotImplemented

    def __pow__(self, p):
        if np.isscalar(p):
            return _PowerLinearOperator(self, p)
        else:
            return NotImplemented

    def __add__(self, x):
        if isinstance(x, LinearOperator):
            return _SumLinearOperator(self, x)
        elif torch.is_tensor(x) and len(x.shape)==2:
            return _SumLinearOperator(self, Lazy(x))
        else:
            return NotImplemented
    def __radd__(self,x):
        return self.__add__(x)
    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, x):
        return self.__add__(-x)

    def __repr__(self):
        M,N = self.shape
        if self.dtype is None:
            dt = 'unspecified dtype'
        else:
            dt = 'dtype=' + str(self.dtype)

        return '<%dx%d %s with %s>' % (M, N, self.__class__.__name__, dt)

    def adjoint(self):
        """Hermitian adjoint.
        Returns the Hermitian adjoint of self, aka the Hermitian
        conjugate or Hermitian transpose. For a complex matrix, the
        Hermitian adjoint is equal to the conjugate transpose.
        Can be abbreviated self.H instead of self.adjoint().
        Returns
        -------
        A_H : LinearOperator
            Hermitian adjoint of self.
        """
        return self._adjoint()

    H = property(adjoint)

    def transpose(self):
        """Transpose this linear operator.
        Returns a LinearOperator that represents the transpose of this one.
        Can be abbreviated self.T instead of self.transpose().
        """
        return self._transpose()

    T = property(transpose)

    def _adjoint(self):
        """Default implementation of _adjoint; defers to rmatvec."""
        return _AdjointLinearOperator(self)

    def _transpose(self):
        """ Default implementation of _transpose; defers to rmatvec + conj"""
        return _TransposedLinearOperator(self)

    def to_dense(self):
        """ Default implementation of to_dense which produces the dense
            matrix corresponding to the given lazy matrix. Defaults to
            multiplying by the identity """
        return self@torch.eye(self.shape[-1])


class _CustomLinearOperator(LinearOperator):
    """Linear operator defined in terms of user-specified operations."""

    def __init__(self, shape, matvec, rmatvec=None, matmat=None,
                 dtype=None, rmatmat=None):
        super(_CustomLinearOperator, self).__init__(dtype, shape)

        self.args = ()

        self.__matvec_impl = matvec
        self.__rmatvec_impl = rmatvec
        self.__rmatmat_impl = rmatmat
        self.__matmat_impl = matmat

        self._init_dtype()

    def _matmat(self, X):
        if self.__matmat_impl is not None:
            return self.__matmat_impl(X)
        else:
            return super(_CustomLinearOperator, self)._matmat(X)

    def _matvec(self, x):
        return self.__matvec_impl(x)

    def _rmatvec(self, x):
        func = self.__rmatvec_impl
        if func is None:
            raise NotImplementedError("rmatvec is not defined")
        return self.__rmatvec_impl(x)

    def _rmatmat(self, X):
        if self.__rmatmat_impl is not None:
            return self.__rmatmat_impl(X)
        else:
            return super(_CustomLinearOperator, self)._rmatmat(X)

    def _adjoint(self):
        return _CustomLinearOperator(shape=(self.shape[1], self.shape[0]),
                                     matvec=self.__rmatvec_impl,
                                     rmatvec=self.__matvec_impl,
                                     matmat=self.__rmatmat_impl,
                                     rmatmat=self.__matmat_impl,
                                     dtype=self.dtype)


class _AdjointLinearOperator(LinearOperator):
    """Adjoint of arbitrary Linear Operator"""
    def __init__(self, A):
        shape = (A.shape[1], A.shape[0])
        super(_AdjointLinearOperator, self).__init__(dtype=A.dtype, shape=shape)
        self.A = A
        self.args = (A,)

    def _matvec(self, x):
        return self.A._rmatvec(x)

    def _rmatvec(self, x):
        return self.A._matvec(x)

    def _matmat(self, x):
        return self.A._rmatmat(x)

    def _rmatmat(self, x):
        return self.A._matmat(x)

class _TransposedLinearOperator(LinearOperator):
    """Transposition of arbitrary Linear Operator"""
    def __init__(self, A):
        shape = (A.shape[1], A.shape[0])
        super(_TransposedLinearOperator, self).__init__(dtype=A.dtype, shape=shape)
        self.A = A
        self.args = (A,)

    def _matvec(self, x):
        # NB. np.conj works also on sparse matrices
        return (self.A._rmatvec((x)))

    def _rmatvec(self, x):
        return (self.A._matvec((x)))

    def _matmat(self, x):
        # NB.  works also on sparse matrices
        return (self.A._rmatmat((x)))

    def _rmatmat(self, x):
        return (self.A._matmat((x)))

def _get_dtype(operators, dtypes=None):
    if dtypes is None:
        dtypes = []
    for obj in operators:
        if obj is not None and hasattr(obj, 'dtype'):
            dtypes.append(obj.dtype)
    return dtypes[0]#removed find_common_dtypes because not supported in jax


class _SumLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or \
                not isinstance(B, LinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        if A.shape != B.shape:
            raise ValueError('cannot add %r and %r: shape mismatch'
                             % (A, B))
        self.args = (A, B)
        super(_SumLinearOperator, self).__init__(_get_dtype([A, B]), A.shape)

    def _matvec(self, x):
        return self.args[0].matvec(x) + self.args[1].matvec(x)

    def _rmatvec(self, x):
        return self.args[0].rmatvec(x) + self.args[1].rmatvec(x)

    def _rmatmat(self, x):
        return self.args[0].rmatmat(x) + self.args[1].rmatmat(x)

    def _matmat(self, x):
        return self.args[0].matmat(x) + self.args[1].matmat(x)

    def _adjoint(self):
        A, B = self.args
        return A.T + B.T

    def invT(self):
        A,B = self.args
        return A.invT() + B.invT()

class _ProductLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or \
                not isinstance(B, LinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        if A.shape[1] != B.shape[0]:
            raise ValueError('cannot multiply %r and %r: shape mismatch'
                             % (A, B))
        super(_ProductLinearOperator, self).__init__(_get_dtype([A, B]),
                                                     (A.shape[0], B.shape[1]))
        self.args = (A, B)

    def _matvec(self, x):
        return self.args[0].matvec(self.args[1].matvec(x))

    def _rmatvec(self, x):
        return self.args[1].rmatvec(self.args[0].rmatvec(x))

    def _rmatmat(self, x):
        return self.args[1].rmatmat(self.args[0].rmatmat(x))

    def _matmat(self, x):
        return self.args[0].matmat(self.args[1].matmat(x))

    def _adjoint(self):
        A, B = self.args
        return B.T * A.T
    
    def to_dense(self):
        A,B = self.args
        A = A.to_dense() if isinstance(A,LinearOperator) else A
        B = B.to_dense() if isinstance(B,LinearOperator) else B
        return A@B


class _ScaledLinearOperator(LinearOperator):
    def __init__(self, A, alpha):
        if not isinstance(A, LinearOperator):
            raise ValueError('LinearOperator expected as A')
        if not np.isscalar(alpha):
            raise ValueError('scalar expected as alpha')
        dtype = _get_dtype([A], [type(alpha)])
        super(_ScaledLinearOperator, self).__init__(dtype, A.shape)
        self.args = (A, alpha)

    def _matvec(self, x):
        return self.args[1] * self.args[0].matvec(x)

    def _rmatvec(self, x):
        return (self.args[1]) * self.args[0].rmatvec(x)

    def _rmatmat(self, x):
        return (self.args[1]) * self.args[0].rmatmat(x)

    def _matmat(self, x):
        return self.args[1] * self.args[0].matmat(x)

    def _adjoint(self):
        A, alpha = self.args
        return A.T * (alpha)
    def to_dense(self):
        A, alpha = self.args
        return alpha*A.to_dense()


class _PowerLinearOperator(LinearOperator):
    def __init__(self, A, p):
        if not isinstance(A, LinearOperator):
            raise ValueError('LinearOperator expected as A')
        if A.shape[0] != A.shape[1]:
            raise ValueError('square LinearOperator expected, got %r' % A)
        if not isinstance(p,int) or p < 0:
            raise ValueError('non-negative integer expected as p')

        super(_PowerLinearOperator, self).__init__(_get_dtype([A]), A.shape)
        self.args = (A, p)

    def _power(self, fun, x):
        res = torch.zeros_like(x)+x#torch.array(x, copy=True)
        for i in range(self.args[1]):
            res = fun(res)
        return res

    def _matvec(self, x):
        return self._power(self.args[0].matvec, x)

    def _rmatvec(self, x):
        return self._power(self.args[0].rmatvec, x)

    def _rmatmat(self, x):
        return self._power(self.args[0].rmatmat, x)

    def _matmat(self, x):
        return self._power(self.args[0].matmat, x)

    def _adjoint(self):
        A, p = self.args
        return A.T ** p


class MatrixLinearOperator(LinearOperator):
    def __init__(self, A):
        super(MatrixLinearOperator, self).__init__(A.dtype, A.shape)
        self.A = A
        self.__adj = None
        self.args = (A,)

    def _matmat(self, X):
        return self.A.dot(X)

    def _adjoint(self):
        if self.__adj is None:
            self.__adj = _AdjointMatrixOperator(self)
        return self.__adj

class _AdjointMatrixOperator(MatrixLinearOperator):
    def __init__(self, adjoint):
        self.A = adjoint.A.T.conj()
        self.__adjoint = adjoint
        self.args = (adjoint,)
        self.shape = adjoint.shape[1], adjoint.shape[0]

    @property
    def dtype(self):
        return self.__adjoint.dtype

    def _adjoint(self):
        return self.__adjoint


class IdentityOperator(LinearOperator):
    def __init__(self, shape, dtype=None):
        super(IdentityOperator, self).__init__(dtype, shape)

    def _matvec(self, x):
        return x

    def _rmatvec(self, x):
        return x

    def _rmatmat(self, x):
        return x

    def _matmat(self, x):
        return x

    def _adjoint(self):
        return self


class Lazy(LinearOperator):
    def __init__(self, dense_matrix):
        self.A = dense_matrix
        super().__init__(None, self.A.shape)
        self._cached_A = None
    
    def _get_A(self, v):
        """Get A matrix on the same device and dtype as v, with caching."""
        target_device = v.device
        target_dtype = v.dtype
        
        if self._cached_A is not None:
            if self._cached_A.device == target_device and self._cached_A.dtype == target_dtype:
                return self._cached_A
        
        self._cached_A = self.A.to(device=target_device, dtype=target_dtype)
        return self._cached_A
        
    def _matmat(self, V):
        return self._get_A(V) @ V
    
    def _matvec(self, v):
        return self._get_A(v) @ v
    
    def _rmatmat(self, V):
        return self._get_A(V).T @ V
    
    def _rmatvec(self, v):
        return self._get_A(v).T @ v
    
    def to_dense(self):
        return self.A



def isintlike(x):
    return isinstance(x,int)


def isshape(x, nonneg=False):
    """Is x a valid 2-tuple of dimensions?
    If nonneg, also checks that the dimensions are non-negative.
    """
    try:
        # Assume it's a tuple of matrix dimensions (M, N)
        (M, N) = x
    except Exception:
        return False
    else:
        if isintlike(M) and isintlike(N):
            if not nonneg or (M >= 0 and N >= 0):
                return True
        return False


import numpy as np
from functools import reduce

product = lambda c: reduce(lambda a,b:a*b,c)

def lazify(x):
    if isinstance(x,LinearOperator): return x
    elif torch.is_tensor(x): return Lazy(x)
    else: raise NotImplementedError

def densify(x):
    if isinstance(x,LinearOperator): return x.to_dense()
    elif torch.is_tensor(x): return x
    else: raise NotImplementedError

class I(LinearOperator):
    def __init__(self,d):
        shape = (d,d)
        super().__init__(None, shape)
    def _matmat(self,V): #(c,k)
        return V
    def _matvec(self,V):
        return V
    def _adjoint(self):
        return self

class LazyKron(LinearOperator):
    """Kronecker product of linear operators.
    
    For matrices A (m×n) and B (p×q), the Kronecker product A⊗B is (mp×nq).
    This class represents the Kronecker product without materializing it.
    
    The key insight for efficient computation:
    (A⊗B)vec(X) = vec(BXA^T) for appropriately shaped X
    """
    def __init__(self, Ms):
        self.Ms = Ms
        shape = product([Mi.shape[0] for Mi in Ms]), product([Mi.shape[1] for Mi in Ms])
        super().__init__(None, shape)

    def _matvec(self, v):
        return self._matmat(v.unsqueeze(1)).squeeze(1) if v.dim() == 1 else self._matmat(v).reshape(-1)
    
    def _matmat(self, v):
        """Efficient Kronecker product computation using reshape and batched matmul."""
        # For 2 matrices A⊗B with shapes (m,n) and (p,q):
        # Input v has shape (n*q,) or (n*q, k)
        # Output has shape (m*p,) or (m*p, k)
        
        if v.dim() == 1:
            v = v.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        
        k = v.shape[1]  # batch dimension
        
        # Reshape v to tensor form: (n1, n2, ..., k) where ni = Mi.shape[1]
        input_shapes = [Mi.shape[-1] for Mi in self.Ms]
        ev = v.reshape(*input_shapes, k)
        
        # Apply each matrix along its corresponding axis
        for i, M in enumerate(self.Ms):
            # Move axis i to front, apply M, move back
            ev = torch.movedim(ev, i, 0)
            original_shape = ev.shape
            # Flatten all dims except first into batch
            ev_flat = ev.reshape(M.shape[-1], -1)
            # Apply M: (m, n) @ (n, batch) -> (m, batch)
            Mev = M @ ev_flat
            # Reshape back
            new_shape = (M.shape[0],) + original_shape[1:]
            ev = Mev.reshape(new_shape)
            ev = torch.movedim(ev, 0, i)
        
        # Reshape to output: (m1*m2*..., k)
        result = ev.reshape(self.shape[0], k)
        
        if squeeze:
            result = result.squeeze(1)
        
        return result
    
    def _adjoint(self):
        return LazyKron([Mi.T for Mi in self.Ms])
    
    def __new__(cls, Ms):
        if len(Ms) == 1:
            return Ms[0]
        return super().__new__(cls)


class ConcatLazy(LinearOperator):
    """ Produces a linear operator equivalent to concatenating
        a collection of matrices Ms along axis=0 """
    def __init__(self,Ms):
        self.Ms = Ms
        assert all(M.shape[1]==Ms[0].shape[1] for M in Ms),\
             f"Trying to concatenate matrices of different sizes {[M.shape for M in Ms]}"
        shape = (sum(M.shape[0] for M in Ms),Ms[0].shape[1])
        super().__init__(None,shape)
    def _matvec(self,v):
        return self._matmat(v)
    def _matmat(self,V):
        return torch.cat([M@V for M in self.Ms],dim=0)
    # def _rmatmat(self,V):
    #     Vs = torch.split(V,len(self.Ms),dim=0)
    #     return sum([self.Ms[i].T@Vs[i] for i in range(len(self.Ms))])
    

class LazyPerm(LinearOperator):
    def __init__(self, perm):
        self.perm = perm
        shape = (len(perm), len(perm))
        super().__init__(None, shape)
        self._cached_perm = None

    def _get_perm(self, v):
        """Get permutation on the same device as v, with caching."""
        target_device = v.device
        
        if self._cached_perm is not None and self._cached_perm.device == target_device:
            return self._cached_perm
        
        self._cached_perm = self.perm.to(target_device)
        return self._cached_perm

    def _matmat(self, V):
        return V[self._get_perm(V)]
    
    def _matvec(self, V):
        return V[self._get_perm(V)]
    
    def _adjoint(self):
        return LazyPerm(torch.argsort(self.perm))

def lazy_direct_matmat(v, Ms, mults):
    """Efficient block-diagonal matrix-vector multiplication.
    
    For a block diagonal matrix with blocks M1, M2, ... (each with multiplicity),
    computes the product with vector v by processing each block separately.
    """
    # Handle 1D vector case
    if v.dim() == 1:
        v = v.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    k = v.shape[1]  # number of columns
    
    results = []
    col_offset = 0
    
    for M, multiplicity in zip(Ms, mults):
        block_cols = multiplicity * M.shape[-1]
        block_rows = multiplicity * M.shape[0]
        
        # Extract the relevant portion of v for this block
        v_block = v[col_offset:col_offset + block_cols]
        
        if multiplicity == 1:
            # Simple case: just apply M
            result_block = M @ v_block
        else:
            # Multiple copies of M on the diagonal
            # Reshape v_block to (M.shape[-1], multiplicity * k)
            v_reshaped = v_block.reshape(multiplicity, M.shape[-1], k).permute(1, 0, 2).reshape(M.shape[-1], -1)
            # Apply M
            Mv = M @ v_reshaped
            # Reshape back to (multiplicity * M.shape[0], k)
            result_block = Mv.reshape(M.shape[0], multiplicity, k).permute(1, 0, 2).reshape(-1, k)
        
        results.append(result_block)
        col_offset += block_cols
    
    output = torch.cat(results, dim=0)
    
    if squeeze_output:
        output = output.squeeze(1)
    
    return output


class LazyDirectSum(LinearOperator):
    """Block diagonal linear operator.
    
    Represents a block diagonal matrix with blocks Ms[0], Ms[1], ...
    Each block can have a multiplicity (repeated on diagonal).
    """
    def __init__(self, Ms, multiplicities=None):
        # Convert numpy arrays to torch tensors
        processed_Ms = []
        for M in Ms:
            if isinstance(M, np.ndarray):
                # Keep original dtype, don't force float32
                processed_Ms.append(torch.from_numpy(M))
            else:
                processed_Ms.append(M)
        self.Ms = processed_Ms
        self._cached_Ms = None  # Cache for device/dtype converted matrices
        
        self.multiplicities = [1 for M in Ms] if multiplicities is None else multiplicities
        shape = (sum(Mi.shape[0] * c for Mi, c in zip(self.Ms, self.multiplicities)),
                 sum(Mi.shape[1] * c for Mi, c in zip(self.Ms, self.multiplicities)))
        super().__init__(None, shape)

    def _get_Ms(self, v):
        """Get all block matrices on the same device and dtype as v, with caching."""
        target_device = v.device
        target_dtype = v.dtype
        
        if self._cached_Ms is not None:
            # Check if cache is valid (same device and dtype)
            sample = self._cached_Ms[0]
            if hasattr(sample, 'device'):
                if sample.device == target_device and sample.dtype == target_dtype:
                    return self._cached_Ms
        
        # Convert all matrices to target device/dtype
        cached = []
        for M in self.Ms:
            if isinstance(M, LinearOperator):
                cached.append(M)  # LinearOperators handle their own caching
            elif isinstance(M, torch.Tensor):
                cached.append(M.to(device=target_device, dtype=target_dtype))
            else:
                cached.append(torch.tensor(M, device=target_device, dtype=target_dtype))
        
        self._cached_Ms = cached
        return self._cached_Ms

    def _matvec(self, v):
        return lazy_direct_matmat(v, self._get_Ms(v), self.multiplicities)

    def _matmat(self, v):
        return lazy_direct_matmat(v, self._get_Ms(v), self.multiplicities)
    
    def _adjoint(self):
        return LazyDirectSum([Mi.T for Mi in self.Ms], self.multiplicities)
    
    def invT(self):
        return LazyDirectSum([M.invT() for M in self.Ms], self.multiplicities)
    
    def to_dense(self):
        """Convert to dense block diagonal matrix using PyTorch."""
        Ms_all = [M for M, c in zip(self.Ms, self.multiplicities) for _ in range(c)]
        Ms_dense = []
        for Mi in Ms_all:
            if isinstance(Mi, LinearOperator):
                Ms_dense.append(Mi.to_dense())
            elif isinstance(Mi, torch.Tensor):
                Ms_dense.append(Mi)
            else:
                Ms_dense.append(torch.tensor(Mi))
        return torch.block_diag(*Ms_dense)