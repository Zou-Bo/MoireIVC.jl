module Basics
    using MKL
    import ClassicalOrthogonalPolynomials: laguerrel
    import ArbNumerics


    "Landau level form factor"
    function LandauLevel_Form_factor(n_left::Int64, n_right::Int64, qx::Float64, qy::Float64; τ::Int64=1,
        l::Float64=1.0)

        if τ^2 != 1
            error("τ must be ±1 in form factor calculation")
        end

        qx *= l
        qy *= l
        min, max, qq = n_left < n_right ? (n_left, n_right, (im*qx+τ*qy)/sqrt(2.)) : (n_right, n_left, (im*qx-τ*qy)/sqrt(2.))
        x = (qx^2 + qy^2)/4.0

        f = laguerrel(min, max-min, 2.0x) * exp(-x)

        for i in min+1:max
            f *= qq
            f /= sqrt(i)
        end

        return f

    end


    "(q×q′)l², inputs are in unit of G1 and G2"
    function ql_cross(q1, q2, q′1, q′2)
        return 2π * (q1*q′2 - q′1*q2)
    end


    "Weierstrass zeta"
    wζ(z, tau) = ComplexF64(ArbNumerics.weierstrass_zeta(ArbNumerics.ArbComplex(z), ArbNumerics.ArbComplex(tau)))
    "Weierstrass sigma"
    wσ(z, tau) = ComplexF64(ArbNumerics.weierstrass_sigma(ArbNumerics.ArbComplex(z), ArbNumerics.ArbComplex(tau)))
    "γ2 parameter for a given lattice {a1, a2}"
    function _γ2(a1, a2)
        a1c = a1[1] + im*a1[2]
        a2c = a2[1] + im*a2[2]
        tau = a2c/a1c
        η1 = wζ(0.5, tau)
        γ2 = (η1 - π*0.5/imag(tau)) / 0.5
        return γ2
    end

end