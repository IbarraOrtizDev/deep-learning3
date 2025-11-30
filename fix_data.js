const data =`¿Cuál es la principal estrategia de Cenicafé para obtener variedades resistentes a la roya?,Utilizar diversidad genética y conformar variedades compuestas con progenies diferentes para asegurar resistencia durable.   
¿Cuál fue la primera variedad resistente liberada por Cenicafé?,La variedad Colombia, liberada en 1982.   
¿Qué característica diferencia a las variedades compuestas?,Están formadas por varias progenies que no son idénticas pero comparten atributos agronómicos comunes.
¿Para qué regiones se recomienda la variedad Tabi?,Para sistemas tradicionales con variedades de porte alto y regiones cafeteras en general.
¿Qué ventaja ofrece la variedad Castillo® frente a variedades susceptibles?,Mayor resistencia a la roya, porte intermedio y alta productividad.
¿Cuántos componentes regionales tiene la variedad Castillo®?,Siete componentes regionales.
¿Qué porcentaje de café supremo ofrece Cenicafé 1?,Supera el 70% de granos supremos.
¿Qué diferencia a Castillo® zonales de Castillo® general?,Las zonales tienen niveles más altos de resistencia a la roya según región.
¿Qué regiones cubre Castillo® zona Norte?,Departamentos del norte como Bolívar, Cesar, Santander, La Guajira y otros. 
¿Qué beneficio económico ofrece sembrar variedades resistentes?,Eliminan el costo de aplicar fungicidas para controlar la roya. 
¿Las variedades resistentes controlan la broca?,No, ninguna variedad es resistente a la broca. 
¿Qué enfermedad africana también manejan estas variedades?,La enfermedad de las cerezas del café (CBD). 
¿El uso de semillas de finca afecta la resistencia?,Sí, reduce la diversidad genética necesaria para mantener la resistencia durable. 
¿Qué origen deben tener las semillas?,Deben adquirirse certificadas en Comités de Cafeteros. 
¿Qué porte presenta Cenicafé 1?,Porte tipo Caturra. 
¿Qué porte presenta Tabi?,Porte alto. 
¿Qué porcentaje máximo de granos vanos permiten estas variedades?,Menos del 10%. 
¿La calidad en taza difiere entre Castillo® y Típica?,No, presentan calidades comparables. 
¿Qué zona cubre Castillo® zona Sur?,Departamentos como Cauca, Huila, Tolima y Nariño. 
¿Cuál es un riesgo de sembrar variedades susceptibles?,Pérdidas productivas superiores al 50% en años de La Niña. 
¿Qué densidad se usa para estudios experimentales de productividad?,5.000 árboles por hectárea. 
¿Qué tipo de brotes pueden presentar las variedades compuestas?,Brotes verdes o bronce. 
¿Qué característica de Castillo® mejora el rendimiento?,Alta productividad y tolerancia a CBD. 
¿Cuánto puede ahorrar un departamento sembrando variedades resistentes?,Más de 80 mil millones en departamentos como Antioquia. 
¿Qué evita la diversidad genética en estas variedades?,Que la roya supere rápidamente la resistencia. 
¿Cuál es la vida útil del estudio de estas variedades?,Más de 20 años de investigación respaldan su uso. 
¿Por qué no se deben mezclar semillas de finca con certificadas?,Se pierde la trazabilidad y capacidad de resistencia. 
¿Cuál variedad ofrece mayor resistencia del grupo Castillo®?,Las variedades Castillo® zonales. 
¿Qué porcentaje de germinación garantiza la semilla de la FNC?,Más del 75%. 
¿Qué evidencia respalda que Castillo® mantiene su resistencia?,Más de 30 años sembrada sin perder resistencia a la roy
¿Por qué la calidad de la semilla es esencial?,Porque determina el vigor y sanidad del almácigo y del establecimiento en campo. 
¿Qué humedad debe tener la semilla para almacenamiento?,Entre 11% y 35%, evitando extremos. 
¿Cuánto dura viable la semilla almacenada correctamente?,Hasta 6 meses. 
¿Cuántas semillas contiene 1 kg?,Más de 4.000 semillas. 
¿Cuántos días tarda la germinación con pergamino?,50 a 70 días. 
¿Se recomienda remojar la semilla antes del germinador?,No es necesario remojarla. 
¿Por qué no se debe obtener semilla en la finca?,Porque se pierde diversidad genética y resistencia. 
¿Cuál es el espesor recomendado de arena en el germinador?,20 cm de arena fina. 
¿Cuál es el espesor de la capa de gravilla?,10 cm para asegurar drenaje. 
¿Cuánto dura la etapa de germinador?,75 a 80 días. 
¿Qué enfermedad frecuente afecta el germinador?,Volcamiento o damping-off por Rhizoctonia solani. 
¿Qué biocontrol se usa para el germinador?,Trichoderma harzianum (Tricho-D).`;
const data2 = []
data.split('\n').forEach(line => {
  const fields = line.split('?,');
  const dataObject = {
    pregunta: fields[0].trim() + '?',
  };
  try {
    const dataFinal = fields[1].split('.,');
    dataObject.respuesta = dataFinal[0].trim();
    //dataObject.categoria = dataFinal[1] ? dataFinal[1].trim() : '';
  } catch (error) {
    console.error(`Error processing line: ${line}`);
  }
  console.log(`"${dataObject.pregunta}","${dataObject.respuesta}"`);
});