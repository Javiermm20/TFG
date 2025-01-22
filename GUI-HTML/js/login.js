var boton = document.getElementById("botonLogin");
var inputUsuario = document.getElementById("usuario");
var inputContrasena = document.getElementById("pass");

async function comprobarLogin(usuario, pass) {		
    try {
        const response = await fetch('http://localhost:3000/api/comprobarLogin', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                usuario: usuario,
                pass: pass,
            }),
            credentials: 'same-origin',
        });
      
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
      
        const data = await response.json();
        if(data.idSesion == null){
            Swal.fire({
                title: 'Login',
                text: 'Usuario o contraseña incorrectos',
                icon: 'error',
                confirmButtonText: 'Aceptar'
            });
            inputUsuario.value="";
        }
        else{
            sessionStorage.setItem("nombreUsuario", usuario);
            sessionStorage.setItem("idSesion", data.idSesion);
            
            const miCookie = data.idSesion;
            
            document.cookie = `miCookie=${miCookie};; expires=Thu, 01 Jan 2025 00:00:00 UTC; path=/`;
            console.log(`${miCookie}`);
            const response = await fetch('/gestion.html', {
                method: 'GET',
                headers: {
                    'Cookie': `${miCookie}`
                },
            });

            if (response.ok) {
                const html = await response.text();
                window.location.href = '/gestion.html';
            } else {
                // Si hay un problema con la solicitud, redirigir a la página de error
                //window.location.href = '/index.html';
            }
        }
        
      
    } catch (error) {
        console.error('Error al actualizar puntos:', error);
    }
}

boton.addEventListener("click", function() {
    var usuario = inputUsuario.value;
    var pass = inputContrasena.value;
    console.log(usuario);
    console.log(pass);
    comprobarLogin(usuario, pass);

});

