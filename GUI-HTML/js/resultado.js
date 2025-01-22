var tabla = document.getElementById("tablaListarResultado");
var elementoAEliminar = sessionStorage.getItem('moleculaSeleccionada');
var predicciones = sessionStorage.getItem("prediccion");

if(sessionStorage.getItem("prediccion") == null){
    window.location.href = '/gestion.html';
}

const resultadosRecuperados = JSON.parse(predicciones);

const linksNavPrincipales = document.querySelectorAll('.navPrincipales a');

linksNavPrincipales.forEach(link => {
  link.classList.remove('active'); 
});

linksNavPrincipales[2].classList.add('active');
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

function reordenarFilasPorPrediccion() {
    const filas = tabla.querySelectorAll('tr');
    const arrayFilas = Array.from(filas).slice(1);

    function compararFilas(filaA, filaB) {
        const valorA = parseFloat(filaA.cells[3].textContent);
        const valorB = parseFloat(filaB.cells[3].textContent);

        if (valorA < valorB) {
            return 1;
        } else if (valorA > valorB) {
            return -1;
        } else {
            return 0;
        }
    }

    
    arrayFilas.sort(compararFilas);

    // Limpiar la tabla antes de insertar las filas reordenadas
    tabla.innerHTML = '';
    const thead = document.createElement('thead');
    const tr = document.createElement('tr');

    // Crear las celdas de encabezado (<th>) y establecer su contenido
    const headers = ['Orden', 'Imagen', 'Identificador', 'Predicción'];
    headers.forEach(texto => {
        const th = document.createElement('th');
        th.textContent = texto;
        th.scope = 'col'; // Establecer el atributo scope="col"
        tr.appendChild(th); // Agregar la celda de encabezado a la fila
    });

    // Agregar la fila de encabezado al encabezado (<thead>)
    thead.appendChild(tr);

    // Insertar el encabezado (<thead>) en la tabla
    tabla.appendChild(thead);

    arrayFilas.forEach((fila, index) => {
        const newRow = tabla.insertRow(); // Crear una nueva fila en la tabla

        // Clonar las celdas de la fila original
        for (let i = 0; i < fila.cells.length; i++) {
            const originalCell = fila.cells[i]; // Obtener la celda original
            const newCell = newRow.insertCell(i); // Crear una nueva celda en la fila reordenada

            // Copiar el contenido de la celda original a la nueva celda
            if (i === 0) {
                // Actualizar el índice (orden) de la primera celda de la fila
                newCell.textContent = (index + 1).toString(); // Nuevo índice (empezando desde 1)
            } else if (originalCell.querySelector('img')) {
                // Si la celda original contiene una imagen
                const img = document.createElement('img');
                img.src = originalCell.querySelector('img').src;
                img.alt = 'Imagen';
                newCell.appendChild(img); // Agregar la imagen a la nueva celda
            } else {
                // Si la celda original no contiene una imagen, copiar el texto
                newCell.textContent = originalCell.textContent;
            }
        }
    });
}


async function cargarTabla(){
    var imagenes = await recuperarImagenes();
    const imagenSeleccionada = document.getElementById('imagenSeleccionada');
    const textoImagen = document.getElementById('textoImagen');
    var index = 1;

    imagenes.forEach(function(imagen) {
        if(imagen._id == elementoAEliminar){
            imagenSeleccionada.src = 'data:image/png;base64,' + imagen.datos;
            textoImagen.innerHTML = imagen._id;
        }
        else{
            const newRow = tabla.insertRow();

            const cellOrden = newRow.insertCell(0);
            const cellImagen = newRow.insertCell(1);
            const cellIdentificador = newRow.insertCell(2);
            const cellPrediccion = newRow.insertCell(3);

            const imageUrl = `data:image/png;base64,${imagen.datos}`;
        
            cellOrden.textContent = index;
            index = index + 1;
            cellImagen.innerHTML = `<img src="${imageUrl}" alt="Imagen">`;
            cellIdentificador.textContent = imagen._id;

            resultadosRecuperados.forEach((fila, index) => {
                if (fila.ID_2 === imagen._id) {
            
                    if (Array.isArray(fila.Prediccion) && fila.Prediccion.length > 0) {
                        if(fila.Prediccion[0] < 0){
                            fila.Prediccion[0] = 0;
                        }
                        else if(fila.Prediccion[0] > 1){
                            fila.Prediccion[0] = 1;
                        }
                        cellPrediccion.textContent = fila.Prediccion[0]; // Muestra el primer elemento del array 'Prediccion'
                    } else {
                        cellPrediccion.textContent = 'No hay predicción disponible';
                    }
                }
            });
        }

    });

    reordenarFilasPorPrediccion();
}

cargarTabla();




