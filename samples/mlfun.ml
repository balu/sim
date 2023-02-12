open Printf

let rec even = function
  | 0 -> true
  | 1 -> false
  | n -> odd (n-1)
and odd = function
  | 0 -> false
  | 1 -> true
  | n -> even (n-1)
;;

let () = printf "%B\n" (even 25)
