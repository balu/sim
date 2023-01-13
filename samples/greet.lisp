(defvar *greeting* "Hello")

(defun greet (name)
  (format t "~a ~a.~%" *greeting* name))

(defun greet-from-india (name)
  (let ((*greeting* "Namaste"))
    (greet name)))

(defun greet-from-france (name)
  (let ((*greeting* "Bonjour"))
    (greet name)))

(greet "Me")
(greet-from-india "Rahul")
(greet "You")
(greet-from-france "Kylian")
