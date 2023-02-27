open Printf

let fact n = 0

let g = fact

let fact n = 1

let () = printf "%d\n" (g 3)

let rec fact n = if n <= 1 then 1 else n*fact(n-1)

let g = fact

let fact n = 1

let () = printf "%d\n" (g 3)
