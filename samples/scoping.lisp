(progn
  (defvar *dynamically-scoped-variable* 5)

  (defun f ()
    (+ (g) (let ((*dynamically-scoped-variable* 6))
             (g))))

  (defun h ()
    (+ (g) (let ((*dynamically-scoped-variable* 7))
             (g))))

  (defun g ()
    (* 2 *dynamically-scoped-variable*))


  (format t "Dynamic scoping: ~a ~a~%" (f) (h)))

(let ((lexically-scoped-variable 5))
  (defun f1 ()
    (+ (g1) (let ((lexically-scoped-variable 6))
              (g1))))

  (defun h1 ()
    (+ (g1) (let ((lexically-scoped-variable 7))
              (g1))))

  (defun g1 ()
    (* 2 lexically-scoped-variable))

  (format t "Lexical scoping: ~a ~a~%" (f1) (h1)))
