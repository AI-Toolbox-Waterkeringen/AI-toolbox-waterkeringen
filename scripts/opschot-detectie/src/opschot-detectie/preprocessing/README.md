# Preprocessing

Op twee dijktrajecten zijn dronevluchten uitgevoerd om hoge resolutie orthofoto's te verzamelen:

* **Hoorn–Enkhuizen (traject 12-2)**
Grote delen van dit traject is verlaagd met V&Z gepenetreerd breuksteen. Voor het noordelijke deel zijn vooral basaltzuilen en blokstenen gebruikt in de bekleding.
* **Medemblik–Den Oever (traject 13-7)**
Een deel van dit traject is Noorse steen te vinden met veel begroeiing. Verder is basalton aanwezig met minder begroeiing.

De RGB-beelden zijn aangeleverd als rasters (.tif bestanden). Vanwege de bestandsgrootte is er een opdeling gemaakt in 12 delen voor traject Hoorn-Enkhuizen en 14 delen voor Medemblik-Den Oever. Daarnaast is een shapefile toegeleverd van beide dijktrajecten.

Tot slot zijn nog twee havenhoofden ingetekend op het traject 12-2. Hier is veel opschot aanwezig en daarom een goede test voor de algoritmes.

## Tegels maken

Voor de input van de modellen halen we kleine tegels uit de rasterbestanden. We gebruiken de geometrie van dijktrajecten over overlappende tegels te kiezen. De grootte van de tegels is in te stellen in het configuratiebestand (`config/config.yaml`), waar `tile_size_x` het aantal pixels in horizontale richting en `tile_size_y` het aantal pixels in verticale richting aangeeft. Wij kiezen voor beide een waarde van 2048.

Om de tegels te genereren kan het volgende script gedraaid worden:

```bash
    python create_tiles.py
```

Tegels die overlappen met de twee havenhoofden kunnen met het volgende script worden gegenereerd:

```bash
    python create_tiles_havenhoofden.py
```
In het configuratiebestand kunnen de shapefiles voor de dijktrajecten worden aangepast.

Het bestand "map_info.csv" geeft het aantal tegels per dijktraject weer.

**Traject Hoorn-Enkhuizen**

De onderstaande tabel geeft een overzicht van gegenereerde tegels voor de 12 delen van traject Hoorn-Enkhuizen.

<div align="left">

|**Deel**|**Naam**|**Aantal tegels**|
|---|---|---|
|1|20230714 MUG Hoorn Enkhuizen orthomosaic deel 1|5|
|2|20230714 MUG Hoorn Enkhuizen orthomosaic deel 2|9|
|3|20230714 MUG Hoorn Enkhuizen orthomosaic deel 3|14|
|4|2v0230714 MUG Hoorn Enkhuizen orthomosaic deel 4|86|
|5|20230714 MUG Hoorn Enkhuizen orthomosaic deel 5|84|
|6|20230714 MUG Hoorn Enkhuizen orthomosaic deel 6|200|
|7|20230714 MUG Hoorn Enkhuizen orthomosaic deel 7|133|
|8|20230714 MUG Hoorn Enkhuizen orthomosaic deel 8|120|
|9|20230714 MUG Hoorn Enkhuizen orthomosaic deel 9|142|
|10|20230714 MUG Hoorn Enkhuizen orthomosaic deel 10|93|
|11|20230714 MUG Hoorn Enkhuizen orthomosaic deel 11|40|
|12|20230714 MUG Hoorn Enkhuizen orthomosaic deel 12|47|
</div>

**Medemblik-Den Oever**

De onderstaande tabel geeft een overzicht van gegenereerde tegels voor de 14 delen van traject Medemblik-Den Oever.

<div align="left">

|**Deel**|**Naam**|**Aantal tegels**|
|---|---|---|
|1|20230714 MUG Medemblik Den Oever orthomosaic deel 1|220|
|2|20230714 MUG Medemblik Den Oever orthomosaic deel 2|252|
|3|20230714 MUG Medemblik Den Oever orthomosaic deel 3|319|
|4|20230714 MUG Medemblik Den Oever orthomosaic deel 4|331|
|5|20230714 MUG Medemblik Den Oever orthomosaic deel 5|147|
|6|20230714 MUG Medemblik Den Oever orthomosaic deel 6|48|
|7|20230714 MUG Medemblik Den Oever orthomosaic deel 7|27|
|8|20230714 MUG Medemblik Den Oever orthomosaic deel 8|22|
|9|20230714 MUG Medemblik Den Oever orthomosaic deel 9|239|
|10|20230714 MUG Medemblik Den Oever orthomosaic deel 10|225|
|11|20230714 MUG Medemblik Den Oever orthomosaic deel 11|188|
|12|20230714 MUG Medemblik Den Oever orthomosaic deel 12|253|
|13|20230714 MUG Medemblik Den Oever orthomosaic deel 13|108|
|14|20230714 MUG Medemblik Den Oever orthomosaic deel 14|78|
| | |

</div>
      </td>
    </tr>
  </table>

</div>


## Example tile Hoorn-Enkhuizen

<img src="../images/voorbeeld_tegel_hoorn_enkhuizen.jpeg" width="600"/>
