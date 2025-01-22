
const linksNavPrincipales = document.querySelectorAll('.navPrincipales a');

linksNavPrincipales.forEach(link => {
  link.classList.remove('active'); 
});

linksNavPrincipales[0].classList.add('active');
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



async function agregarFilaGestion(molecula, usuario){
  try {
    const response = await fetch('http://localhost:3000/api/agregarMoleculaGestion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            molecula: molecula,
            usuario: usuario
        }),
    });
  
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }


    const data = await response.json();
    cargaTotal();

  
  } catch (error) {
      console.error('Error al actualizar puntos:', error);
  }

}

async function borrarFilaGestion(molecula, usuario){
  try {
    const response = await fetch('http://localhost:3000/api/eliminarMoleculaGestion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            molecula: molecula,
            usuario: usuario
        }),
    });
  
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }


    const data = await response.json();
    cargaTotal();

  
  } catch (error) {
      console.error('Error al actualizar puntos:', error);
  }
}

async function cargarDatosImagenes(){
  try {
    const response = await fetch('http://localhost:3000/api/recuperarGestion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            usuario: "root"
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

async function cargarTodosDatos(){
  try {
    const response = await fetch('http://localhost:3000/api/recuperarTodos', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    });
  
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.moleculasTotales;

  
  } catch (error) {
      console.error('Error al actualizar puntos:', error);
  }
}

function cargarTablaComparativa(moleculasSimples, moleculasTotales) {
  var tabla = document.getElementById("tablaGestion");
  var i = 1;

  // Recorre cada molécula en el array `moleculasTotales`
  moleculasTotales.forEach(function(molecula) {
    // Crea una nueva fila en la tabla
    var fila = tabla.insertRow();

    // Inserta celdas para cada columna de la fila
    var celdaNumero = fila.insertCell();
    var celdaIdentificador = fila.insertCell();
    var celdaNombre = fila.insertCell();
    var celdaFecha = fila.insertCell();
    var celdaExisteEnSimples = fila.insertCell(); // Nueva celda para indicar existencia en `moleculasSimples`
    var celdaAcciones = fila.insertCell();

    // Asigna los valores de la molécula a las celdas correspondientes
    celdaNumero.innerText = i;
    celdaIdentificador.innerText = molecula._id;
    celdaNombre.innerText = molecula.denominacion;
    celdaFecha.innerText = molecula.tiempo;

    // Verifica si el identificador de la molécula existe en `moleculasSimples`
    var existeEnSimples = moleculasSimples.some(function(simple) {
      return simple._id === molecula._id;
    });

    // Muestra "Sí" si existe en `moleculasSimples`, de lo contrario muestra "No"
    var existeTexto = existeEnSimples ? "Sí" : "No";
    celdaExisteEnSimples.innerText = existeTexto;

    // Incrementa el contador
    i++;

    if (existeTexto === "No") {
      // Crear icono verde de añadir
      var iconoAgregar = document.createElement('i');
      iconoAgregar.className = "material-icons";
      iconoAgregar.innerText = "add";

      // Crear enlace de añadir
      var enlaceAgregar = document.createElement('a');
      enlaceAgregar.href = "#";
      enlaceAgregar.title = "Add";
      enlaceAgregar.className = "add";

      enlaceAgregar.setAttribute("data-toggle", "tooltip");
      enlaceAgregar.appendChild(iconoAgregar);

      // Agregar evento de clic al enlace de añadir
      enlaceAgregar.addEventListener('click', function() {
        agregarFilaGestion(molecula._id, sessionStorage.getItem("nombreUsuario"));
      });

      // Limpiar celda de acciones y agregar enlace de añadir
      celdaAcciones.innerHTML = '';
      celdaAcciones.appendChild(enlaceAgregar);
    } else {
      // Crear icono rojo de eliminar
      var iconoBorrar = document.createElement('i');
      iconoBorrar.className = "material-icons";
      iconoBorrar.innerText = "delete";

      // Crear enlace de eliminar
      var enlaceBorrar = document.createElement('a');
      enlaceBorrar.href = "#";
      enlaceBorrar.className = "delete";
      enlaceBorrar.title = "Delete";
      enlaceBorrar.setAttribute("data-toggle", "tooltip");
      enlaceBorrar.appendChild(iconoBorrar);

      // Agregar evento de clic al enlace de eliminar
      enlaceBorrar.addEventListener('click', function() {
        borrarFilaGestion(molecula._id, sessionStorage.getItem("nombreUsuario"));
      });

      // Limpiar celda de acciones y agregar enlace de eliminar
      celdaAcciones.innerHTML = '';
      celdaAcciones.appendChild(enlaceBorrar);
    }
  });

  // Código para filtrar las filas según el texto ingresado en el buscador
  var searchInput = document.getElementById('buscadorGestion');

  searchInput.addEventListener('input', function() {
    var searchText = searchInput.value.trim().toLowerCase();

    var filasGestion = document.querySelectorAll('tr');

    for (var i = 1; i < filasGestion.length; i++) {
      var filaGestion = filasGestion[i];
      var identificador = filaGestion.cells[1].innerText.trim().toLowerCase();

      if (identificador.includes(searchText)) {
        filaGestion.style.display = '';
      } else {
        filaGestion.style.display = 'none';
      }
    }
  });
}



function limpiarTabla(tablaId) {
  // Obtener la referencia a la tabla por su ID
  var tabla = document.getElementById(tablaId);

  // Obtener el número de filas en la tabla
  var rowCount = tabla.rows.length;

  // Iterar de abajo hacia arriba sobre las filas de la tabla y eliminar cada una
  for (var i = rowCount - 1; i > 0; i--) {
    tabla.deleteRow(i);
  }
}


async function cargaInicial(){
  limpiarTabla("tablaGestion");
  var moleculas = await cargarDatosImagenes();
  cargarTablaGestion(moleculas);

}

async function cargaTotal(){
  limpiarTabla("tablaGestion");
  var moleculasTotales = await cargarTodosDatos();
  var moleculasSimples = await cargarDatosImagenes();
  cargarTablaComparativa(moleculasSimples, moleculasTotales);

}



function cargarTablaGestion(moleculas){
  var tabla = document.getElementById("tablaGestion");
  var i = 1;

  // Recorre cada molécula en el array `moléculas`
  moleculas.forEach(function(molecula) {
      // Crea una nueva fila en la tabla
      var fila = tabla.insertRow();

      // Inserta celdas para cada columna de la fila
      var celdaNumero = fila.insertCell();
      var celdaIdentificador = fila.insertCell();
      var celdaNombre = fila.insertCell();
      var celdaFecha = fila.insertCell();

      // Asigna los valores de la molécula a las celdas correspondientes
      celdaNumero.innerText = i;
      celdaIdentificador.innerText = molecula._id;
      celdaNombre.innerText = molecula.denominacion;
      celdaFecha.innerText = molecula.tiempo;

      // Incrementa el contador
      i++;

  });

  // Código para filtrar las filas según el texto ingresado en el buscador
  var searchInput = document.getElementById('buscadorGestion');

  searchInput.addEventListener('input', function() {
      var searchText = searchInput.value.trim().toLowerCase();

      var filasGestion = document.querySelectorAll('tr');

      for (var i = 1; i < filasGestion.length; i++) {
          var filaGestion = filasGestion[i];
          var identificador = filaGestion.cells[1].innerText.trim().toLowerCase();

          if (identificador.includes(searchText)) {
              filaGestion.style.display = '';
          } else {
              filaGestion.style.display = 'none';
          }
      }
  });
}

cargaInicial();
var checkbox = document.getElementById("checkboxFiltrar");

checkbox.addEventListener("change", function() {
  if (this.checked) {
    columnaSeleccionada.classList.remove("columna-oculto");
    columnaAcciones.classList.remove("columna-oculto");
    cargaTotal();
  } else {
    columnaSeleccionada.classList.add("columna-oculto");
    columnaAcciones.classList.add("columna-oculto");
    cargaInicial();
  }
});