var boton = document.getElementById("botonPrediccion");
var searchInput = document.getElementById('search');

const linksNavPrincipales = document.querySelectorAll('.navPrincipales a');

linksNavPrincipales.forEach(link => {
  link.classList.remove('active'); 
});

linksNavPrincipales[1].classList.add('active');
linksNavPrincipales[2].addEventListener('click', function(event) {
    event.preventDefault();
    if(sessionStorage.getItem("prediccion") != null){
        const destino = event.target.getAttribute('href');
        window.location.href = destino;
    }
    else{
        Swal.fire({
            title: 'Prediccion',
            text: 'Debes seleccionar una molécula antes de ver los resultados',
            icon: 'error',
            confirmButtonText: 'Aceptar'
        });
    }
});

// Obtén una referencia al elemento de la pantalla de carga
const loader = document.getElementById('loader');

// Función para mostrar la pantalla de carga
function showLoader() {
    loader.style.display = 'flex'; // Mostrar el div de carga
}

// Función para ocultar la pantalla de carga
function hideLoader() {
    loader.style.display = 'none'; // Ocultar el div de carga
}

async function recuperarImagenes(){
  var usuario = sessionStorage.getItem("nombreUsuario");
  try {
    const response = await fetch('http://localhost:3000/api/recuperarPrediccion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            usuario: usuario
        }),
    });
  
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.moleculas;

  
  } catch (error) {
      console.error('Error al actualizar puntos:', error);
  }

}

async function mostrarImagenes(){
  var imagenes = await recuperarImagenes();
  sessionStorage.setItem("imagenes", imagenes);
  var galleryDiv = document.getElementById('galeria');

  imagenes.forEach(function(imagen) {
      var imageContainer = document.createElement('div');
      imageContainer.className = 'image-container';

      var img = document.createElement('img');
      img.src = 'data:image/png;base64,' + imagen.datos;
      img.alt = imagen._id;

      var textoImagen = document.createElement('p');
      textoImagen.textContent = imagen._id; 

      var link = document.createElement('a');

      link.appendChild(img);
      link.appendChild(textoImagen);

      imageContainer.appendChild(link);
      galleryDiv.appendChild(imageContainer);

      link.addEventListener('click', function(event) {
          event.preventDefault();

          var allImages = galleryDiv.querySelectorAll('img');
          allImages.forEach(function(img) {
              img.classList.remove('dorado');
          });

          boton.style.display = "block";

          img.classList.add('dorado');
          var idImagen = img.alt;
          sessionStorage.setItem('moleculaSeleccionada', idImagen);
          
          //window.location.href = link.href; // Redirige a la página de destino
      });
  });

  boton.addEventListener('click', async function(event) {
    showLoader();
    var moleculaSeleccionada = sessionStorage.getItem('moleculaSeleccionada');
    try {
        const respuestaPrediccion = await fetch('http://localhost:3000/api/realizarPrediccion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                imagenes: imagenes,
                moleculaSeleccionada: moleculaSeleccionada
            }),
        });
      
        if (!respuestaPrediccion.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const resultados = await respuestaPrediccion.json();

        const resultadosJSON = JSON.stringify(resultados);

        sessionStorage.setItem("prediccion", resultadosJSON);

        console.log('Resultados:', resultados);

        resultados.forEach(resultado => {
            console.log(`ID_1: ${resultado.ID_1}, ID_2: ${resultado.ID_2}, Predicción: ${resultado.Prediccion}`);
        });
        hideLoader();
        window.location.href = '/resultado.html';

      } catch (error) {
          console.error('Error al actualizar puntos:', error);
      }


    });

  searchInput.addEventListener('input', function() {
      var searchText = searchInput.value.toLowerCase(); // Convertir texto a minúsculas

      // Ocultar o mostrar los contenedores de imagen según el texto de búsqueda
      var containers = document.querySelectorAll('.image-container');
      containers.forEach(function(container) {
          var caption = container.querySelector('p').textContent.toLowerCase(); // Obtener el texto de la leyenda en minúsculas
          if (caption.includes(searchText)) {
              container.style.display = 'block'; // Mostrar contenedor si el texto coincide
          } else {
              container.style.display = 'none'; // Ocultar contenedor si el texto no coincide
          }
      });
  });


}


mostrarImagenes();
